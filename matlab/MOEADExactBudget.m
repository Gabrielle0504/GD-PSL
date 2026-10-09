classdef MOEADExactBudget < ALGORITHM
%MOEADEXACTBUDGET PlatEMO MOEA/D with an exact final FE boundary.
%   Selection, variation, decomposition, and replacement match PlatEMO's
%   MOEAD. The only additional guard stops the final inner loop when maxFE
%   is reached, because UniformPoint may revise N to a non-divisor of maxFE.

    methods
        function main(Algorithm,Problem)
            type = Algorithm.ParameterSet(1);

            [W,Problem.N] = UniformPoint(Problem.N,Problem.M);
            T = ceil(Problem.N/10);
            B = pdist2(W,W);
            [~,B] = sort(B,2);
            B = B(:,1:T);

            Population = Problem.Initialization();
            Z = min(Population.objs,[],1);

            while Algorithm.NotTerminated(Population)
                for i = 1 : Problem.N
                    if Problem.FE >= Problem.maxFE
                        break;
                    end
                    P = B(i,randperm(size(B,2)));
                    Offspring = OperatorGAhalf(Problem,Population(P(1:2)));
                    Z = min(Z,Offspring.obj);

                    switch type
                        case 1
                            normW   = sqrt(sum(W(P,:).^2,2));
                            normP   = sqrt(sum((Population(P).objs-repmat(Z,T,1)).^2,2));
                            normO   = sqrt(sum((Offspring.obj-Z).^2,2));
                            CosineP = sum((Population(P).objs-repmat(Z,T,1)).*W(P,:),2)./normW./normP;
                            CosineO = sum(repmat(Offspring.obj-Z,T,1).*W(P,:),2)./normW./normO;
                            g_old   = normP.*CosineP + 5*normP.*sqrt(1-CosineP.^2);
                            g_new   = normO.*CosineO + 5*normO.*sqrt(1-CosineO.^2);
                        case 2
                            g_old = max(abs(Population(P).objs-repmat(Z,T,1)).*W(P,:),[],2);
                            g_new = max(repmat(abs(Offspring.obj-Z),T,1).*W(P,:),[],2);
                        case 3
                            Zmax  = max(Population.objs,[],1);
                            g_old = max(abs(Population(P).objs-repmat(Z,T,1))./repmat(Zmax-Z,T,1).*W(P,:),[],2);
                            g_new = max(repmat(abs(Offspring.obj-Z)./(Zmax-Z),T,1).*W(P,:),[],2);
                        case 4
                            g_old = max(abs(Population(P).objs-repmat(Z,T,1))./W(P,:),[],2);
                            g_new = max(repmat(abs(Offspring.obj-Z),T,1)./W(P,:),[],2);
                    end
                    Population(P(g_old>=g_new)) = Offspring;
                end
            end
        end
    end
end
