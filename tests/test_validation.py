import pytest

from mlfromscratch.utils.validation import check_X_y


def test_check_X_y_ok():
    X, y = check_X_y([[1, 2], [3, 4]], [10, 20])
    assert X.shape == (2, 2)
    assert y.shape == (2,)


def test_check_X_raises_on_1d():
    with pytest.raises(ValueError):
        check_X_y([1, 2, 3], [1, 2, 3])
