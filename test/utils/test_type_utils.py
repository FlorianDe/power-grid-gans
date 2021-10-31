from utils.type_utils import check_list_type


def test_check_list_type_with_string_list():
    a = ['a', 'b', 'c']

    assert check_list_type(a, str) is True
    assert check_list_type(a, int) is False


def test_check_list_type_with_int_list():
    a = [1, 2, 3]

    assert check_list_type(a, str) is False
    assert check_list_type(a, int) is True


def test_check_list_type_with_mixed_list():
    a = [1, 2, 'a']

    assert check_list_type(a, str) is False
    assert check_list_type(a, int) is False
