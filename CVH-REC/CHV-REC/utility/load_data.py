import os
import numpy as np
import random as rd
import scipy.sparse as sp
from time import time


def _tokenize_interaction_line(line):
    cleaned = line.strip()
    if not cleaned:
        return []
    cleaned = cleaned.replace('(', ' ').replace(')', ' ').replace(',', ' ')
    return [token for token in cleaned.split() if token]


def _detect_triplet_format(lines):
    parsed_lines = [_tokenize_interaction_line(line) for line in lines if line.strip()]
    parsed_lines = [tokens for tokens in parsed_lines if tokens]
    if not parsed_lines:
        return False

    has_tuple_markers = any(
        (',' in line or '(' in line or ')' in line)
        for line in lines
        if line.strip()
    )
    if has_tuple_markers:
        return True

    if any(len(tokens) > 3 for tokens in parsed_lines):
        return False

    third_token_has_decimal = any(
        len(tokens) >= 3 and ('.' in tokens[2] or 'e' in tokens[2].lower())
        for tokens in parsed_lines
    )
    if third_token_has_decimal:
        return True

    has_repeated_user = False
    seen_users = set()
    for tokens in parsed_lines:
        if len(tokens) < 2:
            continue
        try:
            uid = int(tokens[0])
        except ValueError:
            continue
        if uid in seen_users:
            has_repeated_user = True
            break
        seen_users.add(uid)

    return has_repeated_user and any(len(tokens) >= 3 for tokens in parsed_lines)


def _parse_interaction_file(file_path):
    with open(file_path) as f:
        raw_lines = [line.strip() for line in f.readlines() if line.strip()]

    is_triplet = _detect_triplet_format(raw_lines)

    user_items = {}
    interactions = []
    missing_timestamps = 0
    provided_timestamps = 0
    invalid_lines = 0
    interaction_order = 0

    for raw_line in raw_lines:
        tokens = _tokenize_interaction_line(raw_line)
        if len(tokens) < 2:
            invalid_lines += 1
            continue

        if is_triplet:
            try:
                uid = int(tokens[0])
                iid = int(tokens[1])
            except ValueError:
                invalid_lines += 1
                continue

            timestamp = None
            if len(tokens) >= 3:
                try:
                    timestamp = float(tokens[2])
                    provided_timestamps += 1
                except ValueError:
                    missing_timestamps += 1
            else:
                missing_timestamps += 1

            user_items.setdefault(uid, []).append(iid)
            interactions.append((uid, iid, timestamp, interaction_order))
            interaction_order += 1
        else:
            try:
                uid = int(tokens[0])
            except ValueError:
                invalid_lines += 1
                continue

            parsed_items = []
            for token in tokens[1:]:
                try:
                    parsed_items.append(int(token))
                except ValueError:
                    invalid_lines += 1

            if not parsed_items:
                continue

            user_items.setdefault(uid, []).extend(parsed_items)
            for iid in parsed_items:
                interactions.append((uid, iid, None, interaction_order))
                interaction_order += 1

    return {
        'format': 'triplet' if is_triplet else 'static',
        'user_items': user_items,
        'interactions': interactions,
        'missing_timestamps': missing_timestamps,
        'provided_timestamps': provided_timestamps,
        'invalid_lines': invalid_lines,
    }


def _parse_tag_file(file_path):
    entity_to_tags = {}
    max_tag = -1
    with open(file_path) as f:
        for raw_line in f.readlines():
            if not raw_line.strip():
                continue
            tokens = _tokenize_interaction_line(raw_line)
            if not tokens:
                continue
            try:
                entity_id = int(tokens[0])
            except ValueError:
                continue

            tags = []
            for token in tokens[1:]:
                try:
                    tid = int(token)
                    tags.append(tid)
                    max_tag = max(max_tag, tid)
                except ValueError:
                    continue
            entity_to_tags[entity_id] = tags

    return entity_to_tags, max_tag


def _max_dict_key(d):
    if not d:
        return -1
    return max(d.keys())


def _max_dict_values(d):
    max_value = -1
    for values in d.values():
        if values:
            max_value = max(max_value, max(values))
    return max_value

class Data:
    def __init__(self, path, batch_size):
        self.path = path
        self.batch_size = batch_size

        train_file = path + '/TE.csv'
        test_file = path + '/RS.csv'
        api_tag_file=path+'/api_tags.csv'
        mashup_tag_file=path+'/mashup_tags.csv'

        self.n_users, self.n_items = 0, 0
        self.n_train, self.n_test = 0, 0
        self.n_tags = 0
        self.neg_pools = {}

        self.exist_users = []

        # add a dictionary to store recommendation results.
        self.recommendResult = {}

        train_parse = _parse_interaction_file(train_file)
        test_parse = _parse_interaction_file(test_file)

        self.train_parse = train_parse
        self.test_parse = test_parse

        self.train_items = train_parse['user_items']
        self.test_set = test_parse['user_items']

        self.exist_users = list(self.train_items.keys())

        self.n_users = max(_max_dict_key(self.train_items), _max_dict_key(self.test_set)) + 1
        self.n_items = max(_max_dict_values(self.train_items), _max_dict_values(self.test_set)) + 1

        self.n_train = sum(len(items) for items in self.train_items.values())
        self.n_test = sum(len(items) for items in self.test_set.values())

        self.mashup_items, mashup_max_tag = _parse_tag_file(mashup_tag_file)
        self.api_set, api_max_tag = _parse_tag_file(api_tag_file)
        self.n_tags = max(mashup_max_tag, api_max_tag) + 1
        if self.n_tags < 0:
            self.n_tags = 0

        if train_parse['invalid_lines'] > 0:
            print(f"[Data][WARN] Invalid lines ignored in {train_file}: {train_parse['invalid_lines']}")
        if test_parse['invalid_lines'] > 0:
            print(f"[Data][WARN] Invalid lines ignored in {test_file}: {test_parse['invalid_lines']}")

        self.R = sp.dok_matrix((self.n_users, self.n_items), dtype=np.float32)
        self.R_Item_Interacts = sp.dok_matrix((self.n_items, self.n_items), dtype=np.float32)

        for uid, train_items in self.train_items.items():
            for i in train_items:
                self.R[uid, i] = 1.

    def print_statistics(self):
        print('n_users=%d, n_items=%d' % (self.n_users, self.n_items))
        print('n_interactions=%d' % (self.n_train + self.n_test))
        print('n_train=%d, n_test=%d, sparsity=%.5f' % (self.n_train, self.n_test, (self.n_train + self.n_test)/(self.n_users * self.n_items)))


    # generate a negative pool per user.
    def negative_pool(self):
        t1 = time()
        for u in self.train_items.keys():
            neg_items = list(set(range(self.n_items)) - set(self.train_items[u]))
            pools = [rd.choice(neg_items) for _ in range(200)]
            self.neg_pools[u] = pools

    def sample(self):
        if self.batch_size <= self.n_users:
            users = rd.sample(self.exist_users, self.batch_size)
        else:
            users = [rd.choice(self.exist_users) for _ in range(self.batch_size)]

        def sample_pos_items_for_u(u, num):
            pos_items = self.train_items[u]
            n_pos_items = len(pos_items)
            pos_batch = []
            while True:
                if len(pos_batch) == num: break
                pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
                pos_i_id = pos_items[pos_id]

                if pos_i_id not in pos_batch:
                    pos_batch.append(pos_i_id)
            return pos_batch

        def sample_neg_items_for_u(u, num):
            neg_items = []
            while True:
                if len(neg_items) == num: break
                neg_id = np.random.randint(low=0, high=self.n_items, size=1)[0]
                if neg_id not in self.train_items[u] and neg_id not in neg_items:
                    neg_items.append(neg_id)
            return neg_items


        pos_items, neg_items = [], []

        for u in users:
            pos_items += sample_pos_items_for_u(u, 1)
            neg_items += sample_neg_items_for_u(u, 1)


        return users, pos_items, neg_items


    def sample_all_users_pos_items(self):
        self.all_train_users = []
        self.all_train_pos_items = []
        for u in self.exist_users:
            self.all_train_users += [u] * len(self.train_items[u])
            self.all_train_pos_items += self.train_items[u]

    def epoch_sample(self):
        # sample negative items for each user.
        def sample_neg_items_for_u(u, num):
            neg_items = []
            while True:
                if len(neg_items) == num: break
                neg_id = np.random.randint(low=0, high=self.n_items,size=1)[0]
                if neg_id not in self.train_items[u] and neg_id not in neg_items:
                    neg_items.append(neg_id)
            return neg_items

        neg_items = []
        for u in self.all_train_users:
            neg_items += sample_neg_items_for_u(u,1)

        perm = np.random.permutation(len(self.all_train_users))
        # shuffle users, positives and negatives.
        users = np.array(self.all_train_users)[perm]
        pos_items = np.array(self.all_train_pos_items)[perm]
        neg_items = np.array(neg_items)[perm]
        return users, pos_items, neg_items

    def get_num_users_items(self):
        return self.n_users, self.n_items

    def get_sparsity_split(self):
        try:
            split_uids, split_state = [], []
            lines = open(self.path + '/sparsity.split', 'r').readlines()

            for idx, line in enumerate(lines):
                if idx % 2 == 0:
                    split_state.append(line.strip())
                    # print(line.strip())
                else:
                    split_uids.append([int(uid) for uid in line.strip().split(' ')])
            print('get sparsity split.')

        except Exception:
            split_uids, split_state = self.create_sparsity_split()
            f = open(self.path + '/sparsity.split', 'w')
            for idx in range(len(split_state)):
                f.write(split_state[idx] + '\n')
                f.write(' '.join([str(uid) for uid in split_uids[idx]]) + '\n')
            print('create sparsity split.')
        return split_uids, split_state

    def create_sparsity_split(self):
        all_users_to_test = list(self.test_set.keys())
        user_n_iid = dict()

        # generate a dictionary to store (key=n_iids, value=a list of uid).
        for uid in all_users_to_test:
            train_iids = self.train_items[uid]
            test_iids = self.test_set[uid]

            n_iids = len(train_iids) + len(test_iids)

            if n_iids not in user_n_iid.keys():
                user_n_iid[n_iids] = [uid]
            else:
                user_n_iid[n_iids].append(uid)
        split_uids = list()

        # split the whole user set into four subset.
        temp = []
        count = 1
        n_count = (self.n_train + self.n_test)
        n_rates = 0

        split_state = []
        for idx, n_iids in enumerate(sorted(user_n_iid)):
            temp += user_n_iid[n_iids]
            n_rates += n_iids * len(user_n_iid[n_iids])
            n_count -= n_iids * len(user_n_iid[n_iids])

            if n_rates >= count * 0.25 * (self.n_train + self.n_test):
                split_uids.append(temp)

                state = '#inter per user<=[%d], #users=[%d], #all rates=[%d]' %(n_iids, len(temp), n_rates)
                split_state.append(state)
                print(state)

                temp = []
                n_rates = 0

            if idx == len(user_n_iid.keys()) - 1 or n_count == 0:
                split_uids.append(temp)

                state = '#inter per user<=[%d], #users=[%d], #all rates=[%d]' % (n_iids, len(temp), n_rates)
                split_state.append(state)
                print(state)

        return split_uids, split_state


class TemporalDataset(Data):
    def __init__(self, path, batch_size, interaction_file='auto', timestamp_fallback='line_order'):
        super().__init__(path, batch_size)

        if interaction_file in (None, 'auto'):
            temporal_candidate = os.path.join(path, 'TE_temporal.csv')
            self.interaction_file = 'TE_temporal.csv' if os.path.exists(temporal_candidate) else 'TE.csv'
        else:
            self.interaction_file = interaction_file

        self.timestamp_fallback = timestamp_fallback

        temporal_path = os.path.join(path, self.interaction_file)
        print(f"[TemporalDataset] Using interaction file: {self.interaction_file}")
        if self.interaction_file == 'TE.csv':
            temporal_parse = self.train_parse
        else:
            temporal_parse = _parse_interaction_file(temporal_path)

        self.temporal_parse = temporal_parse
        self.temporal_interactions = self._build_temporal_interactions(temporal_parse)
        self.sort_interactions_by_time()

    def _generate_timestamp(self, interaction_order):
        if self.timestamp_fallback == 'line_order':
            return float(interaction_order)
        if self.timestamp_fallback == 'zero':
            return 0.0
        raise ValueError(
            "timestamp_fallback must be one of {'line_order', 'zero'}. "
            f"Got: {self.timestamp_fallback}"
        )

    def _build_temporal_interactions(self, temporal_parse):
        interactions = temporal_parse['interactions']
        provided_timestamps = temporal_parse['provided_timestamps']
        missing_timestamps = temporal_parse['missing_timestamps']

        if provided_timestamps > 0 and missing_timestamps == 0:
            self.timestamp_method = 'provided_in_dataset'
        elif provided_timestamps > 0 and missing_timestamps > 0:
            self.timestamp_method = f'mixed(provided + generated:{self.timestamp_fallback})'
        else:
            self.timestamp_method = f'generated:{self.timestamp_fallback}'

        print(f"[TemporalDataset] Timestamp generation method: {self.timestamp_method}")

        if missing_timestamps > 0:
            print(
                f"[TemporalDataset][WARN] Missing timestamps: {missing_timestamps}. "
                f"Using {self.timestamp_fallback} fallback."
            )

        if temporal_parse['invalid_lines'] > 0:
            print(f"[TemporalDataset][WARN] Invalid temporal lines ignored: {temporal_parse['invalid_lines']}")

        temporal_interactions = []
        for uid, iid, timestamp, interaction_order in interactions:
            ts = timestamp if timestamp is not None else self._generate_timestamp(interaction_order)
            temporal_interactions.append((uid, iid, float(ts)))

        return temporal_interactions

    @staticmethod
    def _is_sorted_by_time(interactions):
        return all(interactions[i][2] <= interactions[i + 1][2] for i in range(len(interactions) - 1))

    def sort_interactions_by_time(self):
        if not self.temporal_interactions:
            return []

        if not self._is_sorted_by_time(self.temporal_interactions):
            print('[TemporalDataset][WARN] Interactions are not sorted by timestamp. Sorting now.')

        self.temporal_interactions.sort(key=lambda x: (x[2], x[0], x[1]))

        if not self._is_sorted_by_time(self.temporal_interactions):
            raise RuntimeError('Incorrect sorting detected for temporal interactions.')

        return self.temporal_interactions

    def split_into_time_slices(self, window_size):
        if window_size <= 0:
            raise ValueError(f'window_size must be > 0, got {window_size}')

        interactions = self.sort_interactions_by_time()
        if not interactions:
            print('[TemporalDataset] Number of slices: 0')
            print('[TemporalDataset] Edges per slice: []')
            return []

        min_timestamp = interactions[0][2]
        slice_buckets = {}

        for uid, iid, timestamp in interactions:
            slice_id = int((timestamp - min_timestamp) // window_size)
            if slice_id < 0:
                slice_id = 0
            if slice_id not in slice_buckets:
                slice_buckets[slice_id] = set()
            slice_buckets[slice_id].add((uid, iid))

        max_slice_id = max(slice_buckets.keys())
        empty_slices = (max_slice_id + 1) - len(slice_buckets)

        graphs = []
        edges_per_slice = []
        for slice_id in sorted(slice_buckets.keys()):
            graph = sp.dok_matrix((self.n_users, self.n_items), dtype=np.float32)
            for uid, iid in slice_buckets[slice_id]:
                graph[uid, iid] = 1.0

            graphs.append(graph.tocsr())
            edges_per_slice.append(len(slice_buckets[slice_id]))

        if empty_slices > 0:
            print(f'[TemporalDataset][WARN] Empty time slices detected/skipped: {empty_slices}')

        print(f'[TemporalDataset] Number of slices: {len(graphs)}')
        print(f'[TemporalDataset] Edges per slice: {edges_per_slice}')

        return graphs

    def get_time_slices(self, window_size):
        return self.split_into_time_slices(window_size)

