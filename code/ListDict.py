class ListDict(dict):
    def update(self, other, **kwargs):
        for key, value in other.items():
            self.setdefault(key, []).extend(value)


def test_ListDict_update():
    list_dict = ListDict()
    list_dict.update({'a': [1, 2, 3]})
    list_dict.update({'b': [4, 5, 6]})
    assert list_dict == {'a': [1, 2, 3], 'b': [4, 5, 6]}
    list_dict.update({'a': [7, 8, 9]})
    assert list_dict == {'a': [1, 2, 3, 7, 8, 9], 'b': [4, 5, 6]}


def test_ListDict_getitem():
    list_dict = ListDict()
    list_dict.update({'a': [1, 2, 3]})
    list_dict.update({'b': [4, 5, 6]})
    assert list_dict['a'] == [1, 2, 3]
    assert list_dict['b'] == [4, 5, 6]
    try:
        list_dict['c']
    except KeyError as e:
        assert str(e) == "'c'"


def test_ListDict_init():
    list_dict = ListDict({'a': [1, 2, 3], 'b': [4, 5, 6]})
    assert list_dict == {'a': [1, 2, 3], 'b': [4, 5, 6]}
    list_dict = ListDict()
    assert list_dict == {}


def test_ListDict_update_performance1():
    start_time = time.time()
    list_dict = ListDict()
    for _ in range(100000):
        i = np.random.randint(0, 1000)
        list_dict.update({str(i): [i, i + 1, i + 2]})
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Update method took {elapsed_time} seconds")
    assert elapsed_time < 0.4, f"Update method took {elapsed_time} seconds, expected less than 0.4"


def test_ListDict_update_performance2():
    elapsed_time = 0
    for _ in range(100):
        list_dict1 = ListDict()
        list_dict2 = ListDict()
        for _ in range(10000):
            i = np.random.randint(0, 1000)
            list_dict1.update({str(i): [i, i + 1, i + 2]})
            list_dict2.update({str(i): [i, i + 1, i + 2]})

        start_time = time.time()
        list_dict = ListDict()
        list_dict.update(list_dict2)
        elapsed_time += time.time() - start_time

    print(f"Update method took {elapsed_time} seconds")
    assert elapsed_time < 0.4, f"Update method took {elapsed_time} seconds, expected less than 0.4"


if __name__ == '__main__':
    import numpy as np
    import time

    test_ListDict_init()
    test_ListDict_getitem()
    test_ListDict_update()

    test_ListDict_update_performance1()
    test_ListDict_update_performance2()
