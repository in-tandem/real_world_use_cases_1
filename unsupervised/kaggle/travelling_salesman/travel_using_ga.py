import random
from copy import deepcopy
from itertools import chain 


def crossover(sample_a, sample_b):

    # random_swath_begin , random_swath_end = random.randint(1, len(sample_a)//2) , \
    #                             random.randint(len(sample_a)//2, len(sample_a) - 2)

    ## using self assigned swath enders to make things easier
    random_swath_begin, random_swath_end = 8, 16

    sub_sample = sample_a[random_swath_begin: random_swath_end]
    
    sub_sample_b = [i for i in sample_b[1:len(sample_b) -1] if i not in sub_sample]

    child = [sample_a[0]] + [None]* ( len(sample_a) - 2) + [sample_a[0]]

    child[random_swath_begin: random_swath_end] = sub_sample

    if random_swath_begin != 1:
        # count = 0
        
        # for i in range(1,random_swath_begin):
        #     child[i] = sub_sample_b[count]
        #     count+=1

        child[1: random_swath_begin] = sub_sample_b[:random_swath_begin - 1 ]
    
    ## at this point, we have 0:random_swath_begin + len(sub_sample) values ready
    ## now we have to populate another total - random_swath_begin - len(sub_sample) - 1 values

    count = random_swath_end

    for i in range(random_swath_begin -1 , len(sub_sample_b) -1 ):
        if None not in child:
            break
        if sub_sample_b[i] in child: ## this prevents duplicates
            continue

        child[count] = sub_sample_b[i]
        count+=1
        
    # for i in range(random_swath_begin, len(sub_sample_b) -1 ):
    #     child[count] = sub_sample_b[i]
    #     count+=1

    # child[random_swath_end:len(child)-1] = sub_sample_b[random_swath_begin:]
    if None  in child:
        print('somak')
        number_of_none = len(list(filter(lambda x: x is None, child)))
        difference = list(set(list(range(1,38))) - set(child))

        ## for each none in child, take one from the difference set and populate
        for count in range(len(difference)):
            if None not in child:
                break
            item = difference[count]
            index = child.index(None)
            child[index] = item
    return child

def one_order_cross_over(sample_a, sample_b):

    """
        the way cross over works is as follow:

        say we have two samples, sample_A  = 1,2,6,8,9,7 sample_b = 9,5,6,7,8,3

        we take a random swath of characters from sample_a / eg say 2-4 ie 6,8,9
        now we take the remaining right most characters from sample_b starting from 4/ ie 3
        we add it to the child . ie 6,8,9,3

        now we take the remaining characters from sample_b which are not in child and
        shuffling it take random remaining characters
        
        in our example, 2 characters are remaining which belongs to (5,7)
        so our child becomes 5,7,6,8,9,3
        

    """

    ## we do not want to touch the first and last element. hence starts from 1(instead of 0)
    ## and ends with len-2 (instead of len - 1)
    random_swath_begin, random_swath_end = random.randint(1, len(sample_a)//2), random.randint(len(sample_a)//2, len(sample_a )  - 2 )

    child = sample_a[random_swath_begin: random_swath_end] + sample_b[random_swath_end:]

    print('diff: ', list(set(sample_b) - set(child)), ', length: ',len(sample_b)-len(child)) 

    diff_bw_scnd_parent_child = list(set(sample_b) - set(child))
    length_diff_bw_scnd_parent_child = len(set(sample_b))-len(set(child)) - 1

    if len(diff_bw_scnd_parent_child)>0 and len(diff_bw_scnd_parent_child) >= length_diff_bw_scnd_parent_child:
        try:
            diff = random.sample(diff_bw_scnd_parent_child, length_diff_bw_scnd_parent_child) 
        except ValueError as e:
            print(e)
    else:
        diff = []

    characters_remaining = [sample_b[0]] + diff

    return characters_remaining+child

class RecordKeeper(object):

    def __init__(self, number_of_cities, adjacency_matrix):
        """
            Simplistic model where we are maintaining path and scores
            in a sequence with order
        """
        self.number_of_cities = number_of_cities
        self.adjacency_matrix = adjacency_matrix
        self.heuristic_upper_range = self.upper_range()
        self.heuristic_lower_range = self.lower_range()
        self.path_travelled = []
        self.score_for_path_travelled = []

    def lower_range(self):
        lowest_possible_distance = min([i for i in chain.from_iterable(self.adjacency_matrix) if i !=0])
        return lowest_possible_distance * self.number_of_cities

    def upper_range(self):
        highest_possible_distance = max([i for i in chain.from_iterable(self.adjacency_matrix) if i !=0])
        return highest_possible_distance * self.number_of_cities
        
    def check_if_range_crossed(self, distance_travelled):
        """
            this particular optimization is incorrect.
            if we select any value within the range of lowest to highest 
            in every single generation it is finding a combination that
            meets this particular scenario
        """
        return True if self.heuristic_lower_range<=distance_travelled <=(self.heuristic_upper_range/3) else False

    def score(self, path, score):

        self.path_travelled.append(path)

        self.score_for_path_travelled.append(score)

    def get_lowest_score_and_path(self):

        min_index, min_value = min(enumerate(self.score_for_path_travelled), key = lambda x: x[1]) 

        return min_value, self.path_travelled[min_index]

class Population(object):
    
    def __init__(self, number, dna_size, sample, mutation_rate, starting_point, adjacency_matrix):

        self.size = number
        self.sample = sample
        self.dna_size = dna_size
        self.dna = []
        self.mutation_rate = mutation_rate
        self.starting_point = starting_point
        self.adjacency_matrix = adjacency_matrix
        self.record = RecordKeeper(number_of_cities = dna_size, adjacency_matrix = adjacency_matrix)


    def populate(self):

        sample = [i for i in self.sample if i!=self.starting_point]
        ## create a random sample of self.dna_size -2 elements
        ## this is because our starting and ending node has to be same
        
        [ self.dna.append(DNA([self.starting_point]+ random.sample(sample, self.dna_size -2 ) + [self.starting_point])) for _ in  range(self.size) ]

    def natural_selection(self):
        """
            we would like to give our dnas with lowest scores (lower score means lower distance)
            higher chances. one very simple way to achieve the same would be to 
            assign score as 1/dna.fitness

            we would also normalize so that all scores are within 0 - 1

        """

        print('Starting to create a more evolved population')

        mating_pool = []
        
        total_fitness_score = sum([i.fitness for i in self.dna])
        total_fitness_score = total_fitness_score if total_fitness_score > 0 else 1

        for dna in self.dna:

            score = dna.fitness/total_fitness_score
            score = score if score!=0 else 1

            score = round((1/score)*100)

            [mating_pool.append(dna) for i in range(score)] if score > 0 else None

        self.reproduce(mating_pool)
        

    def reproduce(self,mating_pool):
        
        print('size of mating pool: ',len(mating_pool))
        for i in range(self.size):

                
            random_index_1 = random.randint(0,len(mating_pool)-1)
            random_index_2 = random.randint(0,len(mating_pool)-1)
            
            parent1 = mating_pool[random_index_1]
            parent2 = mating_pool[random_index_2]

            child = parent1.mate(parent2)
            print('lenght of child : ',len(child.gene))
            self.dna[i] = child.mutate(self.sample, self.mutation_rate)

    def calculate_fitness(self):

        for item in self.dna:
            item.calculate_fitness(self.adjacency_matrix)
            distance_score = item.fitness
            self.record.score(item.gene, distance_score)

            if self.record.check_if_range_crossed(distance_score):
                return True
            


class DNA(object):
    
    def __init__(self, gene):

        self.gene = gene if gene else []
        self.fitness = 0

    def calculate_fitness(self, adjacency_matrix):
        """

            genes in this particular dna denotes the path
            so if gene is 1,2,3,4 our fitness score would be 
            sum of distance from 1->2->3->4

        """
        
        sum = 0

        print('gene to check fitness is : ', self.gene , ' none present in gene : ', None in self.gene)
        for count in range(len(self.gene)-2):
            
            sum = sum + adjacency_matrix[self.gene[count ] - 1][self.gene[count + 1] -1 ]

        self.fitness = sum


    def mutate(self,sample, mutation_rate):
    
        ## we cannot take from the sample cities, since it is possible that city is repeated
        ## dont touch the starting or ending point ever
        self.gene[random.randint(1, len(self.gene) - 2 )] = self.gene[random.randint(1, len(self.gene) - 2 )]
        return self

    
    def mate(self, another_dna):

        ## dont touch the starting and ending point ever
        return DNA(crossover(self.gene, another_dna.gene))

    

def execute():

    adjacency_matrix = [[]]
    adjacency_matrix[0]= [0,10,15,20]
    adjacency_matrix.append([10,0,35,25])
    adjacency_matrix.append([15,35,0,30])
    adjacency_matrix.append([20,25,30,0])

    number_of_cities = 5 #it needs to return the same point
    sample = list(range(1,5))
    starting_point = 1

    epoch = 10

    record = []

    for i in range(epoch):
       
        population = Population(number = 10, \
                                dna_size = number_of_cities, \
                                sample = sample, \
                                mutation_rate = 0.1, \
                                starting_point = starting_point, \
                                adjacency_matrix = adjacency_matrix)

        population.populate()
        
        print('starting genetic mutations....')
        count = 1

        while True:
            
            ## recalculation fitness scores happens over new mating pool
            flag = population.calculate_fitness()
            
            if flag:
                print('Found phrase. solution found in generation: ', count)
                cost, path = population.record.get_lowest_score_and_path()
                print('Path: ', path , ' cost: ', cost)
                record.append((cost, path))
                break
        
            else:
                
                print('Existing mutation failed. Starting natural selection and cross over' )
                population.natural_selection()
        
                print('Natural selection and cross over completed. proceeding to check again')
        
            count = count + 1

    print('Finding lowest score...')
    cost = min(record, key = lambda x:x[0])

    print('minimum cost is ', cost[0], ' path is ', list(filter(lambda x: x == cost, record))[0][1])

# execute()
