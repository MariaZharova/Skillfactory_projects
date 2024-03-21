def train_test_model(target_table, feature_table, model_type, period, dt):
    """ Обучение и тестирование модели с подбором гиперпараметров и логирование метрик """
    # лизинг только для второй версии модели
    product_list_names = ['ebg','fct',  'loan_inv', 'loan_rev', 'dep', 'lsg']
    product_list_codes = ['m_ebg0', 'm_fct0', 'm_loan_inv0', 'm_loan_rev0', 'm_dep0', 'm_lsg0']
    result_table = pd.DataFrame(columns=['model_type', 'period', 'product', 'auc_train', \
                                         'auc_test', 'auc_val', 'feature_imp', 'dt'])
    
    for i in range(len(product_list_names)):
        product_code = product_list_codes[i]
        product = product_list_names[i]
        print(product)
        data = create_data(target_table, feature_table, model_type, product_code, period_list, dt)
        data = shuffle(data, random_state=42)
        X = data[data.columns.drop(list(data.filter(regex='target')))]
        y = data[f'target_{period}']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)
        train_dataset = Pool(data=X_train, label=y_train)
        test_dataset = Pool(data=X_test, label=y_test)
        val_dataset = Pool(data=X_val, label=y_val)
        
        # early stopping для CatBoost
        OPTUNA_EARLY_STOPING = 10
        class EarlyStoppingExceeded(optuna.exceptions.OptunaError):
            early_stop = OPTUNA_EARLY_STOPING
            early_stop_count = 0
            best_score = None

        def early_stopping_opt(study, trial):
            if EarlyStoppingExceeded.best_score == None:
                EarlyStoppingExceeded.best_score = study.best_value

            if study.best_value < EarlyStoppingExceeded.best_score:
                EarlyStoppingExceeded.best_score = study.best_value
                EarlyStoppingExceeded.early_stop_count = 0
            else:
                if EarlyStoppingExceeded.early_stop_count > EarlyStoppingExceeded.early_stop:
                    EarlyStoppingExceeded.early_stop_count = 0
                    best_score = None
                    raise EarlyStoppingExceeded()
                else:
                    EarlyStoppingExceeded.early_stop_count=EarlyStoppingExceeded.early_stop_count+1
            return
        
        def optuna_cb(trial):
            """ Для подбора гиперпараметров CatBoost """
            model = CatBoostClassifier(
                    # iterations=trial.suggest_int("iterations", 100, 10000), # early stopping
                    learning_rate=trial.suggest_float("learning_rate", 1e-3, 1e-1, log=True),
                    depth=trial.suggest_int("depth", 2, 5),
                    l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1e-8, 100.0, log=True),
                    bootstrap_type=trial.suggest_categorical("bootstrap_type", ["Bayesian"]),
                    random_strength=trial.suggest_float("random_strength", 1e-8, 10.0, log=True),
                    bagging_temperature=trial.suggest_float("bagging_temperature", 0.0, 10.0),
                    od_type=trial.suggest_categorical("od_type", ["IncToDec", "Iter"]),
                    od_wait=trial.suggest_int("od_wait", 10, 50),
                    verbose=False,
                    random_state=42,)
            model.fit(train_dataset)
            score = roc_auc_score(y_test, model.predict_proba(test_dataset)[:, 1])
            return score
            
        def plot_feature_importance(feature_names, feature_importance):
            """ Для сохранения и отрисовки графиков feature importances """
            data = pd.DataFrame({'feature_names':feature_names, \
                               'feature_importance':feature_importance}) \
                               .sort_values('feature_importance', ascending=False)
            
            plt.figure(figsize=(8, 5))
            sns.barplot(x=data['feature_importance'], y=data['feature_names'], color='b', alpha=0.5)
            plt.subplots_adjust(left=0.4)
            plt.title(f'Feature importance for {product}')
            plt.xlabel('feature importance')
            os.makedirs(
                f'part2/v3_non_fin/shap_graphics/{model_type}',
                exist_ok=True
            )
            plt.savefig(''.join(f'part2/v3_non_fin/shap_graphics/{model_type}/{product}_{period}_{dt}.png'))

        study = optuna.create_study()
        try:
            study.optimize(optuna_cb, timeout=60, callbacks=[early_stopping_opt], n_trials=5)
        except EarlyStoppingExceeded:
            print(f'EarlyStopping Exceeded: No new best scores on iters {OPTUNA_EARLY_STOPING}')
    
        model = CatBoostClassifier(
                **study.best_params,
                random_state=42,
                silent=True)
        model.fit(train_dataset)
        # model.save_model(f'part2/v1/models/{model_type}_period{period}_{product}')
        plot_feature_importance(X_val.columns, model.get_feature_importance(val_dataset))
        
        y_train_pred = model.predict_proba(train_dataset)[:, 1]
        y_test_pred = model.predict_proba(test_dataset)[:, 1]
        y_val_pred = model.predict_proba(val_dataset)[:, 1]
    
        auc_train = roc_auc_score(y_train, y_train_pred)
        auc_test = roc_auc_score(y_test, y_test_pred)
        auc_val = roc_auc_score(y_val, y_val_pred)
        f_imp_dict = dict(zip(X_val.columns, model.get_feature_importance(val_dataset)))
        f_imp_dict = dict(sorted(f_imp_dict.items(), key=lambda item: -item[1]))
        result_table = result_table.append({'model_type': model_type,
                             'period': period,
                             'product': product, 
                             'auc_train': auc_train, 
                             'auc_test': auc_test,
                             'auc_val': auc_val,
                             'feature_imp': f_imp_dict, 
                             'dt': dt},
                            ignore_index=True)
    return result_table