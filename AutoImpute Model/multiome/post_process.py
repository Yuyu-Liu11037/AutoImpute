import scipy.io
import pandas as pd

# Load the .mat file
mat_file = scipy.io.loadmat('/workspace/AutoImpute/AutoImpute Model/data/imputed_matrix.mat')
print(mat_file.keys())
data = mat_file['arr']

# Create a DataFrame
df = pd.DataFrame(data)
print(df.head())
print('Start writing')
df.to_csv('/workspace/AutoImpute/AutoImpute Model/data/imputed_matrix.csv', index=False)
