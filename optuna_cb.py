X = data[<список колонок для обучения>]
y = data[<таргет>]

# делим все данные на 3 выборки - для обучения, подбора гиперпараметров (валидационная) и тестирования
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

# это катбустовские форматы хранения данных
train_dataset = Pool(data=X_train, label=y_train)
val_dataset = Pool(data=X_val, label=y_val)
test_dataset = Pool(data=X_test, label=y_test)

# early stopping для CatBoost
OPTUNA_EARLY_STOPING = 10
class EarlyStoppingExceeded(optuna.exceptions.OptunaError):
    early_stop = OPTUNA_EARLY_STOPING
    early_stop_count = 0
    best_score = None

def early_stopping_opt(study, trial):
    """ Вспомогательная функция, была найдена на Stack OverFlow """
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
            # iterations подбирать не надо, т.к. уже есть early stopping
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
    # обучаемся на тренировочной выборке
    model.fit(train_dataset)
    # гиперпараметры подбираем по скору на валидационной, чтобы не было сильного переобучения
    score = roc_auc_score(y_test, model.predict_proba(val_dataset)[:, 1])
    return score
    
def plot_feature_importance(feature_names, feature_importance):
    """ 
    Для сохранения и отрисовки графиков feature importances 
    :param feature_names: названия фичей, можно извлечь 
                          при помощи метода columns
    :param feature_importance: важности фичей, достаём из модели
    """
    data = (
        pd.DataFrame({'feature_names': feature_names, 
                      'feature_importance': feature_importance}) 
        .sort_values('feature_importance', ascending=False)
    )
    
    plt.figure(figsize=(8, 5))
    sns.barplot(x=data['feature_importance'], 
                y=data['feature_names'], 
                color='b', alpha=0.5)
    plt.subplots_adjust(left=0.4)
    plt.title('Feature importance')
    plt.xlabel('feature importance')
    # сохранение катинки в файл
    plt.savefig('feature_importance_graph.png'))

# запускаем процесс подбора
study = optuna.create_study()
try:
    study.optimize(optuna_cb, timeout=60, callbacks=[early_stopping_opt], n_trials=5)
except EarlyStoppingExceeded:
    print(f'EarlyStopping Exceeded: No new best scores on iters {OPTUNA_EARLY_STOPING}')

# обучаем модель на лучших гиперпараметрах
model = CatBoostClassifier(
        **study.best_params,
        random_state=42,
        silent=True)
model.fit(train_dataset)
# сохраняем обученную модель
model.save_model('cb_model.pkl')

# отрисовка графика feature importances
# обычно смотрят на важность признаков на тестовой выборке
plot_feature_importance(X_test.columns, 
                        model.get_feature_importance(test_dataset))

# делаем предсказания и считаем метрики (roc-auc для классификации)
y_train_pred = model.predict_proba(train_dataset)[:, 1]
y_val_pred = model.predict_proba(val_dataset)[:, 1]
y_test_pred = model.predict_proba(test_dataset)[:, 1]

auc_train = roc_auc_score(y_train, y_train_pred)
auc_val = roc_auc_score(y_val, y_val_pred)
auc_test = roc_auc_score(y_test, y_test_pred)
print(auc_train, auc_val, auc_test)
