classdef NSGAIIIExactBudget < ALGORITHM
%NSGAIIIEXACTBUDGET PlatEMO NSGA-III with an exact final FE boundary.
%   The final offspring batch is truncated to the remaining FE count after
%   UniformPoint revises N. All selection and environmental-selection logic
%   follows PlatEMO's NSGAIII implementation.

    methods
        function main(Algorithm,Problem)
            [Z,Problem.N] = UniformPoint(Problem.N,Problem.M);
            Population = Problem.Initialization();
            Zmin = min(Population(all(Population.cons<=0,2)).objs,[],1);

            while Algorithm.NotTerminated(Population)
                remaining = Problem.maxFE - Problem.FE;
                offspringCount = min(Problem.N,remaining);
                evenCount = 2*floor(offspringCount/2);
                if evenCount > 0
                    MatingPool = TournamentSelection(2,evenCount,...
                        sum(max(0,Population.cons),2));
                    Offspring = OperatorGA(Problem,Population(MatingPool));
                else
                    parents = TournamentSelection(2,2,...
                        sum(max(0,Population.cons),2));
                    Offspring = OperatorGAhalf(Problem,Population(parents));
                end
                if offspringCount > evenCount && evenCount > 0
                    parents = TournamentSelection(2,2,...
                        sum(max(0,Population.cons),2));
                    Extra = OperatorGAhalf(Problem,Population(parents));
                    Offspring = [Offspring,Extra]; %#ok<AGROW>
                end
                Zmin = min([Zmin;Offspring(all(Offspring.cons<=0,2)).objs],[],1);
                Population = EnvironmentalSelection([Population,Offspring],...
                    Problem.N,Z,Zmin);
            end
        end
    end
end
