class Tokenizer(object):
    def __init__(self, item_ids, user_ids):
        self.item_ids = item_ids
        self.user_ids = user_ids
        self.num_items = len(item_ids)
        self.num_users = len(user_ids)
        self.sos_index = 0
        self.eos_index = 1
        self.item_id_to_index_mapping = {}
        self.user_id_to_index_mapping = {}
        self.index_to_item_id_mapping = {}
        self.index_to_user_id_mapping = {}
        self._initialize_mappings()

    def _initialize_mappings(self):
        self.item_id_to_index_mapping = {
            item_id: index + 2 for index, item_id in enumerate(self.item_ids)
        }
        self.user_id_to_index_mapping = {
            user_id: index for index, user_id in enumerate(self.user_ids)
        }
        self.index_to_item_id_mapping = {
            index + 2: item_id for index, item_id in enumerate(self.item_ids)
        }
        self.index_to_user_id_mapping = {
            index: user_id for index, user_id in enumerate(self.user_ids)
        }

    def _item_id_to_index(self, item_id):
        return self.item_id_to_index_mapping[item_id]

    def _user_id_to_index(self, user_id):
        return self.user_id_to_index_mapping[user_id]

    def _index_to_item_id(self, index):
        return self.index_to_item_id_mapping[index]

    def _index_to_user_id(self, index):
        return self.index_to_user_id_mapping[index]

    @property
    def sos(self):
        return self.sos_index

    @property
    def eos(self):
        return self.eos_index

    def add_new_item(self, item_id):
        # check if item_id is already in the mapping
        if item_id in self.item_id_to_index_mapping:
            return self.item_id_to_index_mapping[item_id]
        else:
            # add item_id to the mapping
            self.item_id_to_index_mapping[item_id] = self.num_items + 2
            self.index_to_item_id_mapping[self.num_items + 2] = item_id
            self.num_items += 1
            return self.num_items + 2

    def add_new_user(self, user_id):
        # check if user_id is already in the mapping
        if user_id in self.user_id_to_index_mapping:
            return self.user_id_to_index_mapping[user_id]
        else:
            # add user_id to the mapping
            self.user_id_to_index_mapping[user_id] = self.num_users
            self.index_to_user_id_mapping[self.num_users] = user_id
            self.num_users += 1
            return self.num_users

    def encode_items(self, items):
        return [self._item_id_to_index(item) for item in items]

    def encode_item(self, item):
        return self._item_id_to_index(item)

    def encode_users(self, users):
        return [self._user_id_to_index(user) for user in users]

    def encode_user(self, user):
        return self._user_id_to_index(user)

    def decode_items(self, items):
        return [self._index_to_item_id(item) for item in items]

    def decode_item(self, item):
        return self._index_to_item_id(item)

    def decode_users(self, users):
        return [self._index_to_user_id(user) for user in users]

    def decode_user(self, user):
        return self._index_to_user_id(user)
