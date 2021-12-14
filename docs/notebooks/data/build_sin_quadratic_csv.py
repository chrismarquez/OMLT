import pandas as pd

n_samples = 10000
w = 5

x = np.linspace(-2,2,n_samples)    
df = pd.DataFrame(x, columns=['x'])
df['y'] = np.sin(w*x) + x**2

df.to_csv("sin_quadratic.csv")