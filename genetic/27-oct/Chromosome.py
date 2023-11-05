class Chromosome:

    def __init__(self, x, y, bird, rank, aptitude):
        # Main data variables
        self.x = x
        self.y = y
        # Linear ranking position
        self.rank = rank
        # Bird function value (the result of x and y values)
        self.bird = bird
        # Sum of the linear indexes (classification, first until current)
        self.aptitude = aptitude