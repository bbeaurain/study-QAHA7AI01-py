class Chapter_A:
    """"""

    def __init__(self):
        """"""
        self.param_a = 22
        self.param_b = "toto"

    def page_1(self, a, b):
        """"""
        print("inside page_1")
        res = a + b
        return res

    def page_2(self, a, b):
        """"""
        print("inside page_2")
        res = a * b
        return res
