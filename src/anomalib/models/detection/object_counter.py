
class QualityCounter:
    """
    A class used to count and track the number of good and defective items in a production process.

    Attributes:
        num_good (int): The initial number of good items (default is 0).
        num_defect (int): The initial number of defective items (default is 0).
    """

    def __init__(self, num_good=0, num_defect=0):
        """
        Initializes the QualityCounter with the specified number of good and defective items.

        Args:
            num_good (int, optional): The initial number of good items. Defaults to 0.
            num_defect (int, optional): The initial number of defective items. Defaults to 0.
        """
        self._num_good = num_good
        self._num_defect = num_defect

    def add_good(self, num=1):
        """
        Increases the count of good items by a specified number.

        Args:
            num (int, optional): The number of good items to add. Defaults to 1.
        """
        self._num_good += num

    def add_defect(self, num=1):
        """
        Increases the count of defective items by a specified number.

        Args:
            num (int, optional): The number of defective items to add. Defaults to 1.
        """
        self._num_defect += num

    @property
    def num_good(self):
        """
        Returns the current count of good items.

        Returns:
            int: The number of good items counted.
        """
        return self._num_good

    @property
    def num_defect(self):
        """
        Returns the current count of defective items.

        Returns:
            int: The number of defective items counted.
        """
        return self._num_defect

    @property
    def total(self):
        """
        Returns the total count of items (both good and defective).

        Returns:
            int: The total number of items.
        """
        return self._num_good + self._num_defect

    def clear(self):
        """
        Resets the counts of both good and defective items to zero.
        """
        self._num_good = 0
        self._num_defect = 0
