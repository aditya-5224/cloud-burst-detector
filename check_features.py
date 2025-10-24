import joblib

model = joblib.load('models/trained/random_forest_model.pkl')
print('Model expects these features in this exact order:')
print('=' * 50)
for i, feat in enumerate(model.feature_names_in_):
    print(f'{i+1:2d}. {feat}')
print('=' * 50)
print(f'Total: {len(model.feature_names_in_)} features')
