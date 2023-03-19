import numpy as np
from tqdm import tqdm

from synthetic.utils import current_time_millis, str_time


def within_tolerance(fitness, best_fitness, tolerance):
    # return abs(fitness - best_fitness) < tolerance
    return max(fitness, best_fitness) / min(fitness, best_fitness) - 1.0 <= tolerance


class EvaluatedIndividual(object):
    def __init__(self, distances_to_net, generator, net):
        self.generator = generator
        self.distances = distances_to_net.compute(net)

        # TODO: other types of fitness
        self.fitness = max(self.distances)
        #self.fitness = (np.array([max(distance, 0.000001) for distance in self.distances]).prod()
        #                ** (1.0 / len(self.distances)))

    def is_better_than(self, eval_indiv, best_fitness, tolerance):
        fitness_orig = self.fitness
        fitness_targ = eval_indiv.fitness

        if tolerance <= 0:
            return fitness_orig < fitness_targ

        if within_tolerance(fitness_orig, best_fitness, tolerance):
            if not within_tolerance(fitness_targ, best_fitness, tolerance):
                return True
            else:
                if self.generator.prog.size() < eval_indiv.generator.prog.size():
                    return True
                elif self.generator.prog.size() == eval_indiv.generator.prog.size():
                    return fitness_orig < fitness_targ
                else:
                    return False
        return False


class Evo(object):
    def __init__(
        self
        , net
        , distances_to_net
        , generations
        , tolerance
        , base_generator
        , out_dir
        , sample_ratio
        , verbosity:int=0
        , save_all:int=0
    ):
        self.distances_to_net = distances_to_net
        self.generations = generations
        self.tolerance = tolerance
        self.base_generator = base_generator
        self.out_dir = out_dir
        self.sample_ratio = sample_ratio

        # number of nodes and edges in target network
        self.nodes = net.graph.vcount()
        self.edges = net.graph.ecount()

        # best individuals
        self.best_individual = None
        self.best_fit_individual = None

        # state
        self.curgen = 0
        self.best_count = 0

        # timers
        self.gen_time = 0
        self.sim_time = 0.
        self.fit_time = 0.
        
        # logs. & replication
        self.verbose_level = verbosity
        self.save_level = save_all

    def run(self):
        # init state
        self.gen_time = 0
        self.sim_time = 0
        self.fit_time = 0
        self.best_count = 0
        self.write_log_header()

        # init population
        generator = self.base_generator.spawn_random()
        net = generator.run(self.nodes, self.edges, self.sample_ratio)
        self.best_fit_individual = EvaluatedIndividual(self.distances_to_net, generator, net)
        self.best_individual = self.best_fit_individual

        # evolve
        saved_best = False
        stable_gens = 0
        self.curgen = 0
        evo_runtime = current_time_millis()
        with tqdm(desc="Stable gens", total=self.generations) as pbar:
            while stable_gens < self.generations:
                self.curgen += 1
                stable_gens += 1

                start_time = current_time_millis()

                self.sim_time = 0
                self.fit_time = 0

                if np.random.randint(0, 2) == 0:
                    generator = self.best_fit_individual.generator.clone()
                else:
                    generator = self.best_individual.generator.clone()

                generator = generator.mutate()

                time0 = current_time_millis()
                net = generator.run(self.nodes, self.edges, self.sample_ratio)
                self.sim_time += current_time_millis() - time0
                time0 = current_time_millis()
                individual = EvaluatedIndividual(self.distances_to_net, generator, net)
                self.fit_time += current_time_millis() - time0

                if individual.is_better_than(self.best_fit_individual, self.best_fit_individual.fitness, 0):
                    self.best_fit_individual = individual
                    stable_gens = 0

                if individual.is_better_than(self.best_individual, self.best_fit_individual.fitness, self.tolerance):
                    self.best_individual = individual
                    self.on_new_best()
                    saved_best = True
                    stable_gens = 0

                # time it took to compute the generation
                self.gen_time = current_time_millis() - start_time
                self.gen_time /= 1000
                self.sim_time /= 1000
                self.fit_time /= 1000
                
                self.on_generation()

                if self.verbose_level > 0:
                    print('stable generation: {}'.format(stable_gens))
                else:
                    if stable_gens == 0:
                        pbar.reset()
                        pbar.set_postfix({
                            '#' : self.best_count,
                            'loss' : self.best_individual.fitness,
                            'size' : self.best_individual.generator.prog.size()
                        })
                    else:
                        pbar.update()
                        
        if not saved_best: # ensure at least one prog & net are saved
            self.on_new_best()

        evo_runtime = current_time_millis()-evo_runtime
        print('Done in {t}. After {n} generations.'.format(t=str_time(evo_runtime/1000), n=self.curgen))
        print("Best found generator:\n\t{}".format(self.best_individual.generator.prog))
        print("\n"+"="*40+"\n")

    def on_new_best(self):
        suffix = '{}_gen{}'.format(self.best_count, self.curgen)
        best_gen = self.best_individual.generator

        # write net --> always save best + store others depending on param
        if self.save_level > 0:
            best_gen.net.graph.save('{}/bestnet{}.gml'.format(self.out_dir, suffix))
        best_gen.net.graph.save('{}/bestnet.gml'.format(self.out_dir))

        # write progs --> always save best + store others depending on param
        if self.save_level > 0:
            best_gen.prog.write('{}/bestprog{}.txt'.format(self.out_dir, suffix))
        best_gen.prog.write('{}/bestprog.txt'.format(self.out_dir))

        self.best_count += 1

    def write_log_header(self):
        # write header of log file
        with open('{}/evo.csv'.format(self.out_dir), 'w') as log_file:
            columns = [
                'gen',
                'best_fit',
                'best_geno_size',
                'lowest_fit',
                'lowest_fit_geno_size',
                'gen_comp_time',
                'sim_comp_time',
                'fit_comp_time',
            ]
            header = ','.join(columns)
            #header = 'gen,best_fit,best_geno_size,gen_comp_time,sim_comp_time,fit_comp_time'
            if hasattr(self.distances_to_net, "targ_stats_set"):
                stat_names = [stat_type.name for stat_type in self.distances_to_net.targ_stats_set.stat_types]
            else: 
                stat_names = []
            header = '{h},{stats_best},{stats_best_fit}\n'.format(
                h=header,
                stats_best=','.join(stat_names),
                stats_best_fit='best_fit'+',best_fit'.join(stat_names),
            )
            log_file.write(header)

    def on_generation(self, force_print:bool=False):
        best_dists = [str(dist) for dist in self.best_individual.distances]
        best_fit_dists = [str(dist) for dist in self.best_fit_individual.distances]

        # write log line for generation
        with open('{}/evo.csv'.format(self.out_dir), 'a') as log_file:
            row = ','.join([str(metric) for metric in (self.curgen,
                            self.best_individual.fitness,
                            self.best_individual.generator.prog.size(),
                            self.best_fit_individual.fitness,
                            self.best_fit_individual.generator.prog.size(),
                            self.gen_time, self.sim_time, self.fit_time)])
            row = '{},{},{}\n'.format(row, ','.join(best_dists), ','.join(best_fit_dists))
            log_file.write(row)

        # print info --> if verbosity > 0
        if self.verbose_level > 0 or force_print:
            print('>>> GENERATION #{}; gen comp time: {}s.; sim comp time: {}s.; fit comp time: {}s.'.format(
                self.curgen, self.gen_time, self.sim_time, self.fit_time))
            if hasattr(self.distances_to_net, "targ_stats_set"):
                stat_names = [stat_type.name for stat_type in self.distances_to_net.targ_stats_set.stat_types]
            else: 
                stat_names = []
            print('[BEST GENERATOR] fitness: {}; genotype size: {}'.format(
                self.best_individual.fitness, self.best_individual.generator.prog.size()))
            print('; '.join(['{}: {}'.format(stat_names[i], best_dists[i]) for i in range(len(stat_names))]))
            print('[LOWEST FITNESS] fitness: {}; genotype size: {}'.format(
                self.best_fit_individual.fitness, self.best_fit_individual.generator.prog.size()))
            print('; '.join(['{}: {}'.format(stat_names[i], best_fit_dists[i]) for i in range(len(stat_names))]))