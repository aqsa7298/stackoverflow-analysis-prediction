import numpy as np
import pandas as pd
import re
import time
import matplotlib.pyplot as plt
np.warnings.filterwarnings('error', category=np.VisibleDeprecationWarning)
pd.options.mode.chained_assignment = None  # default='warn'

def read_df(csv_list):
    
    df_list = []
    # read multiple csv files and add them in a single dataframe
    for filename in csv_list:
        df_list.append(pd.read_csv(filename, index_col = 'Id', parse_dates = ['CreationDate']))     
    data = pd.concat(df_list)

    return data

# function to preprocess the dataframe
def preprocess_df(df):
    
    s1 = time.time()
    count_tags = []
    all_tags = []
    all_tags_df = []
    tag_nump = np.array(df['Tags'])
    # get all tags and their counts. append them in separate lists
    for row in tag_nump:
        val = re.findall(r'\<(.*?)\>', row)
        all_tags_df.append(val)
        all_tags.extend(val)
        count_tags.append(len(val))
    
    e1 = time.time()
    print('Time for preprosessing loop is:   ', e1-s1)
    
    all_tags_df_tuple = tuple(tuple(x) for x in all_tags_df)
    # add new column 'Total Tags' having count of tags against each row    
    df['Total_Tags'] = count_tags
    # df['Tags List'] = all_tags_df
    df['Tags_List'] = np.array(all_tags_df_tuple, dtype=object)
    #convert tags to numpy array for easier maipulation (performance)
    all_tags_array = np.array(all_tags, dtype=object)
    all_unique_tags , unique_count = np.unique(all_tags_array, return_counts = True)

    # make and sort dictionary comprising of unique tags and their counts
    unique_tags_dict = dict(zip(all_unique_tags, unique_count))
    sorted_tags_count = sorted(unique_tags_dict.items(), key=lambda x: x[1], reverse=True)

    # overwrite 'Body' column with new string after removing <p> and </p> 
    df['Body'] = df['Body'].str.replace('<p>', '').str.replace('</p>', '')

    # add new column 'Title Total Words' having count of total words in Title of each row
    df['Title_Total_Words'] = df['Title'].str.split().str.len()

    # add new column 'Body Total Words' having count of total words in Body of each row
    df['Body_Total_Words'] = df['Body'].str.split().str.len()

    # add new column 'Creation Year'
    df['Creation_Year'] = pd.DatetimeIndex(df['CreationDate']).year

    return df, sorted_tags_count 

# function to get top languages and relevant information
def top_language(df, tags):

    top_tags_name = []
    top_tags_count = []
    c = 0

    # get top 5 tags and their counts. Store them in separate lists 
    while c < 5:
        top_tags_name.append(tags[c][0])
        top_tags_count.append(tags[c][1])
        c+=1    
    top_tags_name_tuple = tuple(top_tags_name)

    # filter dataframe rows which have top 5 programming languages
    mask = df['Tags_List'].apply(lambda x: any( val in x for val in top_tags_name_tuple))
    
    # new dataframe with filtered rows
    df_lang = df[mask]

    start = time.time()
    lang_list = []

    # convert series to numpy array for better performance  
    lang = np.array(df_lang['Tags_List'])
    
    # Most efficient way for loop As doing vectorization i.e. using numpy array instead of series 
    for row in lang:
        for j in row:
            if j in top_tags_name:
                lang_list.append(j)
                break

    df_lang['Language_Tag'] = lang_list
    end = time.time()
    print('Runtime of vectorziation numpy is: ', end-start)


    # Vectorization performance can be compared with other methods mentioned below:
    # Below code uses 4 methods 
    # 1. Apply(). it is less efficient
    # 2. Normal nested for loop: Almost same efficiency as apply() function
    # 3. Zip() function: performs a little better than apply() and for loop
    # 4. itertuples(): performs worst in this scenario since we only need single column for comparison. not whole dataframe

    # In order to see performance difference UNCOMMENT below code. running time for each method will be printed befor ploting graph
    
    
    
    # start1 = time.time()
    # df_lang['Language_Tag'] = df_lang['Tags_List'].apply(lambda x: func(x, top_tags_name))
    # end1 = time.time()
    # print ('The time for using apply() is:   ', end1-start1)

    # start2 = time.time()
    # l_list1 = []
    # for row in df_lang['Tags_List']:
    #     for j in row:
    #         if j in top_tags_name:
    #             l_list1.append(j)
    #             break

    # df_lang['Language_Tag1'] = l_list1
    # end2 = time.time()
    # print('the time for by using normal nested for loop directly on series is:  ', end2-start2)
    
    # start3 = time.time()
    # l_list2 = []
    # for (t,r) in zip(df_lang.index, df_lang['Tags_List']):
    #     for i in r:
    #         if i in top_tags_name:
    #             l_list2.append(i)
    #             break
    # df_lang['Language_Tag2'] = l_list2
    # end3 = time.time()
    # print('Runtime of for loop (zip) is: ', end3-start3)

    # start4 = time.time()
    # l_list3 = []
    # for row in df_lang.itertuples():
    #     for i in row.Tags_List:
    #         if i in top_tags_name:
    #             l_list3.append(i)
    #             break
    # df_lang['Language_Tag3'] = l_list3
    # end4 = time.time()
    # print('Runtime of itertuples() in dataframe is: ', end4-start4)

    return df_lang, top_tags_name, top_tags_count

def plot_graphs_df(csv_file_list):
    
    # csv_file_list = ['./stackoverflow/train.csv', './stackoverflow/valid.csv']
    data = read_df(csv_file_list)
    
    # here tags = dictionary of unique tags and their count
    train_data, tags = preprocess_df(data)
    print('Data after all preprocessing is')
    print (train_data.info())
    print(train_data.head())
    train_data['Y'].hist(color='purple', grid=False)
    plt.title('Questions Types')
    plt.tight_layout()    
    plt.savefig('./images/question_types.png')
    plt.show()
    
    grouped = train_data.groupby(['Creation_Year', 'Y']).size()
    grp = grouped.unstack().plot(kind = 'bar')
    grp.set_title('Questions Types based on Year')
    plt.tight_layout()    
    plt.savefig('./images/question_types_year.png')
    plt.show()

    # plot bar chart for top prpgramming languages
    df_language, top_tags_name, top_tags_count = top_language(train_data, tags)
    plt.bar(top_tags_name, top_tags_count, color ='maroon',width = 0.4)
    plt.title('Top Programming Languages')
    plt.tight_layout()    
    plt.savefig('./images/top_program_lang.png')
    plt.show()

    #plot clustered bar chart to see top programming languages distribution over years
    grouped_language = df_language.groupby(['Creation_Year', 'Language_Tag']).size()
    grp_lang = grouped_language.unstack().plot(kind = 'bar')
    grp_lang.set_title('Top Programming Languages Distribution over Years 2016-2020')
    plt.tight_layout()    
    plt.savefig('./images/program_lang_year.png')
    plt.show()
    
if __name__ == "__main__":
    csv_file_list = ['./stackoverflow/train.csv', './stackoverflow/valid.csv']
    plot_graphs_df(csv_file_list)
