from analysis import plot_graphs_df
from training import preprocess_doc2vec_models, preprocess_tfidf_models

if __name__ == "__main__":
    
    csv_file_list = ['./stackoverflow/train.csv', './stackoverflow/valid.csv']

    print('Some useful data insights')
    plot_graphs_df(csv_file_list)

    print('Predictive Analysis')
    preprocess_tfidf_models(csv_file_list)
    preprocess_doc2vec_models(csv_file_list)
    