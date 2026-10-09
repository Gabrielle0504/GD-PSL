classdef NSGAIIExactBudget < ALGORITHM
%NSGAIIEXACTBUDGET PlatEMO NSGA-II with an exact final FE boundary.
%   Standard generations contain Problem.N offspring. Only the final batch is
%   shortened when fewer evaluations remain, after which the usual NSGA-II
%   nondominated sorting and crowding-distance selection are applied.

    methods
        function main(Algorithm,Problem)
            Population = Problem.Initialization();
            [Population,FrontNo,CrowdDis] = NSGAIIExactBudget.Select(Population,Problem.N);

            while Algorithm.NotTerminated(Population)
                remaining = Problem.maxFE - Problem.FE;
                offspringCount = min(Problem.N,remaining);
                evenCount = 2*floor(offspringCount/2);
                if evenCount > 0
                    MatingPool = TournamentSelection(2,evenCount,FrontNo,-CrowdDis);
                    Offspring = OperatorGA(Problem,Population(MatingPool));
                else
                    parents = TournamentSelection(2,2,FrontNo,-CrowdDis);
                    Offspring = OperatorGAhalf(Problem,Population(parents));
                end
                if offspringCount > evenCount && evenCount > 0
                    parents = TournamentSelection(2,2,FrontNo,-CrowdDis);
                    Extra = OperatorGAhalf(Problem,Population(parents));
                    Offspring = [Offspring,Extra]; %#ok<AGROW>
                end
                [Population,FrontNo,CrowdDis] = NSGAIIExactBudget.Select( ...
                    [Population,Offspring],Problem.N);
            end
        end
    end

    methods (Static, Access = private)
        function [Population,FrontNo,CrowdDis] = Select(Population,N)
            [FrontNo,MaxFNo] = NDSort(Population.objs,Population.cons,N);
            Next = FrontNo < MaxFNo;
            CrowdDis = CrowdingDistance(Population.objs,FrontNo);
            Last = find(FrontNo == MaxFNo);
            [~,Rank] = sort(CrowdDis(Last),'descend');
            Next(Last(Rank(1:N-sum(Next)))) = true;
            Population = Population(Next);
            FrontNo = FrontNo(Next);
            CrowdDis = CrowdDis(Next);
        end
    end
end
