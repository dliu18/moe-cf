import numpy as np

class lastfm:
    def load_ratings(self, data_dir):
        num_users = 1892
        num_artists = 17632
        self.P = np.zeros((num_users, num_artists))

        self.user_id_to_idx = {}
        self.artist_id_to_idx = {}

        self.user_idx_to_id = {}
        self.artist_idx_to_id = {}

        curr_user_idx = 0
        curr_artist_idx = 0

        with open(data_dir + "user_artists.dat", "r") as dataFile:
            for line in (dataFile.readlines())[1:]:
                row = line.strip().split("\t")
                user_id = row[0]
                artist_id = row[1]
                weight = int(row[2])

                user_idx, artist_idx = (-1, -1)
                if user_id in self.user_id_to_idx:
                    user_idx = self.user_id_to_idx[user_id]
                else:
                    user_idx = curr_user_idx 
                    self.user_id_to_idx[user_id] = user_idx
                    self.user_idx_to_id[user_idx] = user_id
                    curr_user_idx += 1
                if artist_id in self.artist_id_to_idx:
                    artist_idx = self.artist_id_to_idx[artist_id]
                else:
                    artist_idx = curr_artist_idx 
                    self.artist_id_to_idx[artist_id] = artist_idx
                    self.artist_idx_to_id[artist_idx] = artist_id
                    curr_artist_idx += 1

                self.P[user_idx][artist_idx] = weight

        assert curr_user_idx == num_users
        assert curr_artist_idx == num_artists
    
    def load_metadata(self, data_dir):
        self.artist_id_to_name = {}
        with open(data_dir + "artists.dat", "r") as dataFile:
            for line in dataFile.readlines()[1:]:
                artist_info = line.split("\t")
                self.artist_id_to_name[artist_info[0].strip()] = artist_info[1].strip()
                
        self.tags_per_artist = {}
        with open(data_dir + "user_taggedartists.dat", "r") as dataFile:
            for line in dataFile.readlines()[1:]:
                tag_info = line.split("\t")
                artist_id = tag_info[1]
                tag_id = tag_info[2]

                if artist_id not in self.tags_per_artist:
                    self.tags_per_artist[artist_id] = {}
                if tag_id not in self.tags_per_artist[artist_id]:
                    self.tags_per_artist[artist_id][tag_id] = 0

                self.tags_per_artist[artist_id][tag_id] += 1

        tag_to_artist = {}
        tag_threshold = 10
        for artist_id in self.tags_per_artist:
            for tag_id in self.tags_per_artist[artist_id]:
                if self.tags_per_artist[artist_id][tag_id] > tag_threshold:
                    if tag_id not in tag_to_artist:
                        tag_to_artist[tag_id] = []
                    tag_to_artist[tag_id].append(artist_id)
        min_artists_per_tag = 1
        tags_with_sufficient_artists = []
        for tag in tag_to_artist:
            if len(tag_to_artist[tag]) > min_artists_per_tag:
                tags_with_sufficient_artists.append(tag)
        self.tag_to_artist = {tag: tag_to_artist[tag] for tag in tags_with_sufficient_artists}

        self.tag_id_to_name = {}
        with open(data_dir + "tags.dat", "r",  encoding='latin') as dataFile:
            for line in dataFile.readlines()[1:]:
                tag_info = line.split("\t")
                self.tag_id_to_name[tag_info[0].strip()] = tag_info[1].strip()
                
    def get_P(self):
        return self.P
    
    def get_tag_id_from_name(self, tagname):
        for tag_id in self.tag_id_to_name:
            if self.tag_id_to_name[tag_id] == tagname:
                return tag_id
        return -1
    
    def get_tagnames(self, min_artist = 0, max_artist = 0):
        tags_with_sufficient_artists = []
        for tag in self.tag_to_artist.keys():
            if len(self.tag_to_artist[tag]) >= min_artist and len(self.tag_to_artist[tag]) <= max_artist:
                tags_with_sufficient_artists.append(
                    self.tag_id_to_name[tag]
                )
        return tags_with_sufficient_artists
    
    def __init__(self, data_dir = "data/lastfm/"):
        self.load_ratings(data_dir)
        self.load_metadata(data_dir)
    
    def filter(self, min_users=50, min_ratings=20):
        print("Shape before filtering: ", self.P.shape)
        users_to_keep_idx = []
        users_to_keep_id = []
        for i in range(len(self.P)):
            if np.sum(self.P[i, :] > 0) >= min_ratings:
                users_to_keep_idx.append(i)
                users_to_keep_id.append(self.user_idx_to_id[i])
        self.P = self.P[users_to_keep_idx, :]
        self.users_to_keep_idx = np.array(users_to_keep_idx)
        self.users_to_keep_id = np.array(users_to_keep_id)

        artists_to_keep_idx = []
        artists_to_keep_ids = []
        for j in range(self.P.shape[1]):
            if np.sum(self.P[:, j] > 0) >= min_users:
                artists_to_keep_idx.append(j)
                artists_to_keep_ids.append(self.artist_idx_to_id[j])
        self.P = self.P[:, artists_to_keep_idx]
        self.artists_to_keep_idx = np.array(artists_to_keep_idx)
        self.artists_to_keep_ids = np.array(artists_to_keep_ids)

        users_with_interactions = np.sum(self.P, axis=1) > 0
        self.P = self.P[users_with_interactions]
        self.users_to_keep_idx = self.users_to_keep_idx[users_with_interactions]
        self.users_to_keep_id = self.users_to_keep_id[users_with_interactions]

        print("Shape after filtering: ", self.P.shape)
        return self.P
        
    def kdd_filter(self, min_users=50, min_ratings=20):
        print("Shape before filtering: ", self.P.shape)
        artists_to_keep_idx = []
        artists_to_keep_ids = []
        for j in range(self.P.shape[1]):
            if np.sum(self.P[:, j] > 0) > min_users:
                artists_to_keep_idx.append(j)
                artists_to_keep_ids.append(self.artist_idx_to_id[j])
        self.P = self.P[:, artists_to_keep_idx]
        self.artists_to_keep_idx = np.array(artists_to_keep_idx)
        self.artists_to_keep_ids = np.array(artists_to_keep_ids)

        users_to_keep_idx = []
        users_to_keep_id = []
        for i in range(len(self.P)):
            if np.sum(self.P[i, :] > 0) > min_ratings:
                users_to_keep_idx.append(i)
                users_to_keep_id.append(self.user_idx_to_id[i])
        self.P = self.P[users_to_keep_idx, :]
        self.users_to_keep_idx = np.array(users_to_keep_idx)
        self.users_to_keep_id = np.array(users_to_keep_id)

        users_with_interactions = np.sum(self.P, axis=1) > 0
        self.P = self.P[users_with_interactions]
        self.users_to_keep_idx = self.users_to_keep_idx[users_with_interactions]
        self.users_to_keep_id = self.users_to_keep_id[users_with_interactions]

        print("Shape after filtering: ", self.P.shape)
        return self.P

    def print_tags(self):
        for tag_id in self.tag_to_artist:
            print("{}: {}".format(self.tag_id_to_name[tag_id],
                                 [self.artist_id_to_name[artist_id] for artist_id in self.tag_to_artist[tag_id]]))
            print("\n")
            
    def users_by_tag(self, tagnames, exclusive = True):
        """
            Filter the matrix P such that each horizontal submatrix corresponds to a tag.
            Users belong to a tag if their top artist is exclusively in the tag
        """
        tags = [self.get_tag_id_from_name(tagname) for tagname in tagnames]
        
        def is_exclusive_tag(artist_id, query_tag, tags):
            assert np.all([tag in self.tag_to_artist for tag in tags])
            if artist_id not in self.tag_to_artist[query_tag]:
                return False
            
            for tag in tags:
                if tag == query_tag:
                    continue
                if artist_id in self.tag_to_artist[tag]:
                    return False
            return True
        
        idxs_per_tag = {tag: [] for tag in tags}


        #tag id for pop is 24 and tag id for rock is 73
        for i in range(len(self.P)):
            fav_artist_idx = np.argmax(self.P[i]) #this is the idx in P not in the original matrix
            fav_artist_id = self.artists_to_keep_ids[fav_artist_idx]
            for tag in tags:
                if exclusive and is_exclusive_tag(fav_artist_id, tag, tags):
                    idxs_per_tag[tag].append(i)
                    continue
                
                if not exclusive and fav_artist_id in self.tag_to_artist[tag]:
                    idxs_per_tag[tag].append(i)

        combined_idxs = []
        idxs_per_tag_output = {}
        idx_counter = 0
        for tag in idxs_per_tag:
            n = len(idxs_per_tag[tag])
            combined_idxs.extend(idxs_per_tag[tag])
            idxs_per_tag_output[tag] = np.arange(idx_counter, idx_counter + n)
            idx_counter += n
        
        return self.P[combined_idxs], idxs_per_tag_output
    
    
    def convert_artist_id_to_name(self, artist_id):
        return self.artist_id_to_name[artist_id]
    
    def convert_artist_idx_to_id(self, artist_idx):
        return self.artists_to_keep_ids[artist_idx]
    
    def convert_artist_idx_to_name(self, artist_idx):
        artist_id = self.convert_artist_idx_to_id(artist_idx)
        return self.convert_artist_id_to_name(artist_id)

if __name__ == "__main__":
    obj = lastfm()
    obj.filter()
    #obj.print_tags()
    sub_P, idxs = obj.users_by_tag(["pop", "rock"])
    print(sub_P.shape)
    print(len(idxs[obj.get_tag_id_from_name("pop")]))
    print(len(idxs[obj.get_tag_id_from_name("rock")]))