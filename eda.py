import matplotlib.pyplot as plt
import seaborn as sns

def plot_outcome_distribution(df):
    sns.countplot(x='outcome', data=df)
    plt.title("Launch Success vs Failure")
    plt.xlabel("Outcome (1=Success, 0=Failure)")
    plt.ylabel("Count")
    plt.show()
