import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from math import pi

sns.set_theme(style="whitegrid")


def MakeDF(data, Blockname):
    block = len(data)
    method_name = data[0].columns
    box_per_block = len(method_name)
    len_of_box = len(data[0].index)

    MethodName = []
    mName = []
    for i in method_name:
        for j in range(len_of_box):
            mName.append(i)

    for i in range(block):
        MethodName += mName

    DataName = []
    for i in range(block):
        for j in range(box_per_block * len_of_box):
            DataName.append(Blockname[i])

    X = []
    for i in range(block):
        X.append([])

    for k in range(block):
        for i in range(box_per_block):
            for j in range(len_of_box):
                X[k].append(data[k][data[k].columns[i]].iloc[j])

    DataX = []
    for i in range(block):
        DataX += X[i]

    dfn = pd.DataFrame({
        "data": DataX,
        "Method": MethodName,
        "step": DataName
    })
    
    return dfn



def SpiderPlot(df, yticks, ylabel, ylim, label, col, path):
    # ------- PART 1: Create background
    
    # number of variable
    categories=list(df)
    N = len(categories)

    # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    # Initialise the spider plot
    ax = plt.subplot(111, polar=True)

    # If you want the first axis to be on top:
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)

    
    # ------- PART 2: Add plots

    # Plot each individual = each line of the data
    # I don't make a loop, because plotting more than 3 groups makes the chart unreadable

    # Ind1 df.shape[1]-1
    for i in range(df.shape[0]):
        values=df.iloc[i].values.flatten().tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=3, linestyle='solid', label=label[i])
        #ax.fill(angles, values, col[i], alpha=0.1)

    # Draw one axe per variable + add labels
    plt.xticks(angles[:-1], categories, size=10)
    
    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks(yticks, ylabel, color="grey", size=10)
    plt.ylim(ylim[0],ylim[1])
    

    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.0, 0.3))
    plt.savefig(path, dpi = 300, format = 'eps', bbox_inches='tight')
    # Show the graph
    plt.show()