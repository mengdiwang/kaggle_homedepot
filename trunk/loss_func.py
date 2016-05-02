__author__ = 'Mdwang'


from sklearn.metrics import mean_squared_error, make_scorer
# RMSE score methord


def fmean_squared_error(ground_truth, predictions):
    fmean_squared_error_ = mean_squared_error(ground_truth, predictions)**0.5
    return fmean_squared_error_

RMSE = make_scorer(fmean_squared_error, greater_is_better=False)


# TODO: too slow, use filter
def check_valid(df_sol, y_pred_valid, public=True):
    sol = []
    pred = []
    if public:
        for index, row in df_sol.iterrows():
            if row['Usage'] == "Public":
                sol.append(row['relevance'])
                pred.append(y_pred_valid[index])

    else:
        for index, row in df_sol.iterrows():
            if row['Usage'] == "Private":
                sol.append(row['relevance'])
                pred.append(y_pred_valid[index])

    return fmean_squared_error(sol, pred)