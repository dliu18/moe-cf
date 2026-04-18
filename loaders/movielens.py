import numpy as np
from tqdm import tqdm
import pickle

class movielens:
    def load_raw_ratings(self, data_dir):
        # load ratings matrix
        with open(data_dir + "ml-1m/ratings.dat", "r") as ratingsFile:
            user_id_to_idx = {}
            movie_id_to_idx = {}

            user_idx = 0
            movie_idx = 0
            for line in ratingsFile.readlines():
                line_entries = line.split("::")
                assert len(line_entries) == 4
                user_id, movie_id, rating, _ = line_entries
                if user_id not in user_id_to_idx:
                    user_id_to_idx[user_id] = user_idx
                    user_idx += 1
                if movie_id not in movie_id_to_idx:
                    movie_id_to_idx[movie_id] = movie_idx
                    movie_idx += 1

            n = user_idx
            m = movie_idx
            
        with open(data_dir + "ml-1m/ratings.dat", "r") as ratingsFile:
            ratings = np.zeros((n, m))
            for line in ratingsFile.readlines():
                line_entries = line.split("::")
                assert len(line_entries) == 4
                user_id, movie_id, rating, _ = line_entries
                ratings[user_id_to_idx[user_id], movie_id_to_idx[movie_id]] = int(rating)
        return ratings, movie_id_to_idx, user_id_to_idx
        
    def get_idx_to_id(self, id_to_idx):
        idx_to_id = {}
        for id_key in id_to_idx:
            idx_to_id[id_to_idx[id_key]] = id_key
        return idx_to_id

    def read_user_labels(self, data_dir):
        '''
        Datafile format:
        userId::Gender::Age::Occupation::Zipcode
        '''
        self.user_labels = {
            "Gender": {gender: [] for gender in ["M", "F"]}, 
            "Age": {age: [] for age in ["1", "18", "25", "35", "45", "50", "56"]}
        }
        with open(data_dir + "ml-1m/users.dat", "r") as usersFile:
            for line in usersFile.readlines():
                features = line.split("::")
                assert len(features) == 5

                user_id, gender, age, _, _ = features
                assert gender in self.user_labels["Gender"] and age in self.user_labels["Age"]

                self.user_labels["Gender"][gender].append(user_id)
                self.user_labels["Age"][age].append(user_id)

    
    def __init__(self, min_ratings = 0, min_users = 0, binary = True, data_dir = "data/"):
        self.data_dir = data_dir
        
        raw_ratings, movie_id_to_idx, user_id_to_idx = self.load_raw_ratings(data_dir)
        self.read_user_labels(data_dir)
        print("Shape before filtering: ", raw_ratings.shape)

        movie_idx_to_id = self.get_idx_to_id(movie_id_to_idx)
        user_idx_to_id = self.get_idx_to_id(user_id_to_idx)

        users_to_keep_idx = []
        for i in range(len(raw_ratings)):
            if np.sum(raw_ratings[i, :] > 0) >= min_ratings:
                users_to_keep_idx.append(i)
        print(len(users_to_keep_idx))
        raw_ratings = raw_ratings[users_to_keep_idx]

        items_to_keep_idx = []
        for j in range(raw_ratings.shape[1]):
            if np.sum(raw_ratings[:, j] > 0) >= min_users:
                items_to_keep_idx.append(j)

        ratings_filtered = raw_ratings[:, items_to_keep_idx]
        self.movie_idx_to_id = {i : movie_idx_to_id[items_to_keep_idx[i]] for i in range(len(items_to_keep_idx))}
        self.user_idx_to_id = {i : user_idx_to_id[users_to_keep_idx[i]] for i in range(len(users_to_keep_idx))}

        if binary:
            ratings_classified = np.zeros(ratings_filtered.shape)
            # ratings_classified[ratings_filtered > 0] = -1
            ratings_classified[ratings_filtered > 0] = 1
            self.ratings = ratings_classified
        else:
            self.ratings = ratings_filtered

        print("Shape after filtering: ", self.ratings.shape)

    def get_X(self):
        return self.ratings
    
    def get_movie_metadata(self):
        movie_id_to_metadata = {}
        with open("{}/ml-1m/movies.dat".format(self.data_dir), "r", encoding = "ISO-8859-1") as movieFile:
            for line in movieFile.readlines():
                movie_id, title, genres = line.split("::")
                metadata = {
                    "title": title,
                    "genres": genres.strip().split("|")
                }
                movie_id_to_metadata[movie_id] = metadata
        idx_to_metadata = {idx: movie_id_to_metadata[self.movie_idx_to_id[idx]] for idx in range(len(self.movie_idx_to_id))}
        return idx_to_metadata

    def get_user_labels(self, group_name):
        '''
        returns a mapping from group labels to row indices
        '''
        assert group_name in ["Age", "Gender"]

        output_mapping = {}
        user_id_to_idx = {user_id: idx for idx, user_id in self.user_idx_to_id.items()}
        for group_label in self.user_labels[group_name]:
            user_idxs = [
                user_id_to_idx[user_id] \
                for user_id in self.user_labels[group_name][group_label] \
                if user_id in user_id_to_idx
            ]
            output_mapping[group_label] = user_idxs
        return output_mapping

if __name__ == "__main__":
    movielens_obj = movielens(min_ratings = 0, min_users = 200, binary=True)
    label_to_idxs = movielens_obj.get_user_labels("Age")
    X = movielens_obj.get_X()
    idx_to_metadata = movielens_obj.get_movie_metadata()

    for label, user_idxs in label_to_idxs.items():
        # print(f"{label}: \t {len(user_idxs)}")

        print(f"****** Group {label} ******")
        print(f"Size: {len(user_idxs)}")

        movie_totals = np.sum(X[user_idxs], axis=0)
        for movie_idx in np.argsort(-movie_totals)[:3]:
            metadata = idx_to_metadata[movie_idx]
            print(f"{metadata['title']}\t {movie_totals[movie_idx] / len(user_idxs)}")
        print("\n")

    print(np.min(np.sum(X, axis=0)))
    print(np.min(np.sum(X, axis=1)))  
    print(np.sum(X))  

    # idx_to_genres = movielens_obj.get_genres()
    # assert X.shape[1] == len(idx_to_genres)

    # genres_to_idx = {}
    # for idx in idx_to_genres:
    #     for genre in idx_to_genres[idx]:
    #         if genre not in genres_to_idx:
    #             genres_to_idx[genre] = []
    #         genres_to_idx[genre].append(idx)

    # column_sample = []
    # for genre in genres_to_idx:
    #     idxs = np.array(genres_to_idx[genre])
    #     top_idx = np.argsort(np.sum(X[:, idxs] != 0, axis = 0))[-25:]
    #     new_column_sample = column_sample.copy()
    #     for idx in idxs[top_idx]:
    #         if idx not in column_sample:
    #             new_column_sample.append(idx)
    #     column_sample = new_column_sample

    # print(len(column_sample))
    # X = X[:, column_sample]

    # X_top_movies = np.zeros(X.shape)
    # for i in range(len(X)):
    #     top_movie_idxs = np.argsort(X[i])[-25:]
    #     X_top_movies[i, top_movie_idxs] = X[i, top_movie_idxs]
    # X_top_movies[X_top_movies < 4] = 0
    # X_top_movies[X_top_movies > 0] = 1

    # X = X_top_movies

    # rs = np.arange(1, X.shape[1] + 1)
    # proj_matrices_max_pred = {}
    
    # with open("pickles/proj_matrices_max_pred_movielens_genres.pickle", "wb") as pickleFile:
    #     pickle.dump((X, proj_matrices_max_pred), pickleFile)
    
    # for r in tqdm(rs):
    #     proj_matrices_max_pred[r] = utils.fair_pca_max_pred(X, X.shape[1], r)
    
    #     with open("pickles/proj_matrices_max_pred_movielens_genres.pickle", "wb") as pickleFile:
    #         pickle.dump((X, proj_matrices_max_pred), pickleFile)
