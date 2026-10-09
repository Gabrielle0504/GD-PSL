classdef REProblemBase < PROBLEM
    %REPROBLEMBASE PlatEMO-compatible implementations of the project RE set.
    %
    % The Python definitions use constraint violation as the final objective
    % for the constrained RE benchmarks.  This class preserves that contract
    % so the MATLAB search output can be validated by the Python pipeline.

    methods
        function Population = Evaluation(obj,varargin)
            Population = Evaluation@PROBLEM(obj,varargin{:});
            log_platemo_evaluations(Population);
        end

        function Setting(obj)
            name = upper(class(obj));
            switch name
                case 'RE21'
                    obj.M = 2; obj.D = 4;
                    obj.lower = [1 sqrt(2) sqrt(2) 1];
                    obj.upper = [3 3 3 3];
                case 'RE22'
                    obj.M = 2; obj.D = 3;
                    obj.lower = [0.2 0 0]; obj.upper = [15 20 40];
                case 'RE23'
                    obj.M = 2; obj.D = 4;
                    obj.lower = [1 1 10 10]; obj.upper = [100 100 200 240];
                case 'RE24'
                    obj.M = 2; obj.D = 2;
                    obj.lower = [0.5 0.5]; obj.upper = [4 50];
                case 'RE25'
                    obj.M = 2; obj.D = 3;
                    obj.lower = [1 0.6 0.09]; obj.upper = [70 30 0.5];
                case 'RE31'
                    obj.M = 3; obj.D = 3;
                    obj.lower = [1e-5 1e-5 1]; obj.upper = [100 100 3];
                case 'RE32'
                    obj.M = 3; obj.D = 4;
                    obj.lower = [0.125 0.1 0.1 0.125]; obj.upper = [5 10 10 5];
                case 'RE33'
                    obj.M = 3; obj.D = 4;
                    obj.lower = [55 75 1000 11]; obj.upper = [80 110 3000 20];
                case 'RE34'
                    obj.M = 3; obj.D = 5;
                    obj.lower = ones(1,5); obj.upper = 3*ones(1,5);
                case 'RE35'
                    obj.M = 3; obj.D = 7;
                    obj.lower = [2.6 0.7 17 7.3 7.3 2.9 5];
                    obj.upper = [3.6 0.8 28 8.3 8.3 3.9 5.5];
                case 'RE36'
                    obj.M = 3; obj.D = 4;
                    obj.lower = 12*ones(1,4); obj.upper = 60*ones(1,4);
                case 'RE37'
                    obj.M = 3; obj.D = 4;
                    obj.lower = zeros(1,4); obj.upper = ones(1,4);
                case 'RE41'
                    obj.M = 4; obj.D = 7;
                    obj.lower = [0.5 0.45 0.5 0.5 0.875 0.4 0.4];
                    obj.upper = [1.5 1.35 1.5 1.5 2.625 1.2 1.2];
                case 'RE42'
                    obj.M = 4; obj.D = 6;
                    obj.lower = [150 20 13 10 14 0.63];
                    obj.upper = [274.32 32.31 25 11.71 18 0.75];
                case 'RE61'
                    obj.M = 6; obj.D = 3;
                    obj.lower = [0.01 0.01 0.01]; obj.upper = [0.45 0.1 0.1];
                case 'RE91'
                    obj.M = 9; obj.D = 7;
                    obj.lower = [0.5 0.45 0.5 0.5 0.875 0.4 0.4];
                    obj.upper = [1.5 1.35 1.5 1.5 2.625 1.2 1.2];
                otherwise
                    error('REProblemBase:UnknownProblem','Unsupported RE class %s.',name);
            end
            obj.encoding = ones(1,obj.D);
        end

        function PopObj = CalObj(obj,PopDec)
            x = double(PopDec);
            name = upper(class(obj));
            switch name
                case 'RE21', PopObj = obj.re21(x);
                case 'RE22', PopObj = obj.re22(x);
                case 'RE23', PopObj = obj.re23(x);
                case 'RE24', PopObj = obj.re24(x);
                case 'RE25', PopObj = obj.re25(x);
                case 'RE31', PopObj = obj.re31(x);
                case 'RE32', PopObj = obj.re32(x);
                case 'RE33', PopObj = obj.re33(x);
                case 'RE34', PopObj = obj.re34(x);
                case 'RE35', PopObj = obj.re35(x);
                case 'RE36', PopObj = obj.re36(x);
                case 'RE37', PopObj = obj.re37(x);
                case 'RE41', PopObj = obj.re41(x);
                case 'RE42', PopObj = obj.re42(x);
                case 'RE61', PopObj = obj.re61(x);
                case 'RE91', PopObj = obj.re91(x);
                otherwise, error('REProblemBase:UnknownProblem','Unsupported RE class %s.',name);
            end
        end
    end

    methods (Access=private)
        function v = violation(~, varargin)
            G = cat(2,varargin{:});
            v = sum(max(0,-G),2);
        end

        function y = nearest(~, values, x)
            values = values(:)';
            x = x(:);
            [~,idx] = min(abs(x-values),[],2);
            y = values(idx(:));
            y = y(:);
        end

        function F = re21(obj,x)
            F = 10; E = 2e5; L = 200;
            f1 = L*(2*x(:,1)+sqrt(2)*x(:,2)+sqrt(x(:,3))+x(:,4));
            f2 = F*L/E*(2./x(:,1)+2*sqrt(2)./x(:,2)-2*sqrt(2)./x(:,3)+2./x(:,4));
            F = [f1 f2]; %#ok<NASGU>
        end

        function F = re22(obj,x)
            vals = [0.20 0.31 0.40 0.44 0.60 0.62 0.79 0.80 0.88 0.93 1.0 1.20 1.24 1.32 1.40 1.55 1.58 1.60 1.76 1.80 1.86 2.0 2.17 2.20 2.37 2.40 2.48 2.60 2.64 2.79 2.80 3.0 3.08 3.10 3.16 3.41 3.52 3.60 3.72 3.95 3.96 4.0 4.03 4.20 4.34 4.40 4.65 4.74 4.80 4.84 5.0 5.28 5.40 5.53 5.72 6.0 6.16 6.32 6.60 7.11 7.20 7.80 7.90 8.0 8.40 8.69 9.0 9.48 10.27 11.0 11.06 11.85 12.0 13.0 14.0 15.0];
            x1 = obj.nearest(vals,x(:,1)); x2=x(:,2); x3=x(:,3);
            g1=x1.*x3-7.735*safeDivide(x1.^2,x2)-180;
            g2=4-safeDivide(x3,x2);
            F=[29.4*x1+0.6*x2.*x3 obj.violation(g1,g2)];
        end

        function F = re23(obj,x)
            x1=0.0625*round(x(:,1)); x2=0.0625*round(x(:,2)); x3=x(:,3); x4=x(:,4);
            f1=0.6224*x1.*x3.*x4+1.7781*x2.*x3.^2+3.1661*x1.^2.*x4+19.84*x1.^2.*x3;
            g1=x1-0.0193*x3; g2=x2-0.00954*x3;
            g3=pi*x3.^2.*x4+(4/3)*pi*x3.^3-1296000;
            F=[f1 obj.violation(g1,g2,g3)];
        end

        function F = re24(obj,x)
            x1=x(:,1); x2=x(:,2); E=700000;
            f1=x1+120*x2; sigmaB=4500./(x1.*x2); tau=1800./x2;
            delta=562000./(E*x1.*x2.^2); sigmaK=E*x1.^2/100;
            g1=1-sigmaB/700; g2=1-tau/450; g3=1-delta/1.5; g4=1-sigmaB./sigmaK;
            F=[f1 obj.violation(g1,g2,g3,g4)];
        end

        function F = re25(obj,x)
            vals=[0.009 0.0095 0.0104 0.0118 0.0128 0.0132 0.014 0.015 0.0162 0.0173 0.018 0.02 0.023 0.025 0.028 0.032 0.035 0.041 0.047 0.054 0.063 0.072 0.08 0.092 0.105 0.12 0.135 0.148 0.162 0.177 0.192 0.207 0.225 0.244 0.263 0.283 0.307 0.331 0.362 0.394 0.4375 0.5];
            x1=round(x(:,1)); x2=x(:,2); x3=obj.nearest(vals,x(:,3));
            f1=pi^2*x2.*x3.^2.*(x1+2)/4; Cf=(4*x2./x3-1)./(4*x2./x3-4)+0.615*x3./x2;
            K=11.5e6*x3.^4./(8*x1.*x2.^3); lf=1000./K+1.05*(x1+2).*x3; sigmaP=300./K;
            g1=-(8*Cf*1000.*x2)./(pi*x3.^3)+189000; g2=-lf+14; g3=-3+x2./x3;
            g4=-sigmaP+6; g5=-sigmaP-700./K-1.05*(x1+2).*x3+lf; g6=1.25-700./K;
            F=[f1 obj.violation(g1,g2,g3,g4,g5,g6)];
        end

        function F = re31(obj,x)
            x1=x(:,1);x2=x(:,2);x3=x(:,3); f1=x1.*sqrt(16+x3.^2)+x2.*sqrt(1+x3.^2);
            f2=safeDivide(20*sqrt(16+x3.^2),x3.*x1);
            g1=0.1-f1; g2=100000-f2; g3=100000-safeDivide(80*sqrt(1+x3.^2),x3.*x2);
            F=[f1 f2 obj.violation(g1,g2,g3)];
        end

        function F = re32(obj,x)
            x1=x(:,1);x2=x(:,2);x3=x(:,3);x4=x(:,4);P=6000;L=14;E=30e6;G=12e6;
            f1=1.10471*x1.^2.*x2+0.04811*x3.*x4.*(14+x2); f2=4*P*L^3./(E*x4.*x3.^3);
            M=P*(L+x2/2); R=sqrt(x2.^2/4+((x1+x3)/2).^2); J=2*sqrt(2)*x1.*x2.*(x2.^2/12+((x1+x3)/2).^2);
            tdd=M.*R./J; td=P./(sqrt(2)*x1.*x2); tau=sqrt(td.^2+(2*td.*tdd.*x2)./(2*R)+tdd.^2); sigma=6*P*L./(x4.*x3.^2);
            pc=(4.013*E.*sqrt(x3.^2.*x4.^6/36)/L^2).*(1-(x3/(2*L))*sqrt(E/(4*G)));
            F=[f1 f2 obj.violation(13600-tau,30000-sigma,x4-x1,pc-P)];
        end

        function F = re33(obj,x)
            x1=x(:,1);x2=x(:,2);x3=x(:,3);x4=x(:,4); d=x2.^2-x1.^2;
            f1=4.9e-5*d.*(x4-1); f2=9.82e6*d./(x3.*x4.*(x2.^3-x1.^3));
            g1=x2-x1-20; g2=0.4-x3./(3.14*d); g3=1-2.22e-3*x3.*(x2.^3-x1.^3)./d.^2; g4=2.66e-2*x3.*x4.*(x2.^3-x1.^3)./d-900;
            F=[f1 f2 obj.violation(g1,g2,g3,g4)];
        end

        function F = re34(obj,x)
            x1=x(:,1);x2=x(:,2);x3=x(:,3);x4=x(:,4);x5=x(:,5);
            f1=1640.2823+2.3573285*x1+2.3220035*x2+4.5688768*x3+7.7213633*x4+4.4559504*x5;
            f2=6.5856+1.15*x1-1.0427*x2+0.9738*x3+0.8364*x4-0.3695*x1.*x4+0.0861*x1.*x5+0.3628*x2.*x4-0.1106*x1.^2-0.3437*x3.^2+0.1764*x4.^2;
            f3=-0.0551+0.0181*x1+0.1024*x2+0.0421*x3-0.0073*x1.*x2+0.024*x2.*x3-0.0118*x2.*x4-0.0204*x3.*x4-0.008*x3.*x5-0.0241*x2.^2+0.0109*x4.^2;
            F=[f1 f2 f3];
        end

        function F = re35(obj,x)
            x1=x(:,1);x2=x(:,2);x3=round(x(:,3));x4=x(:,4);x5=x(:,5);x6=x(:,6);x7=x(:,7);
            f1=0.7854*x1.*x2.^2.*(10*x3.^2/3+14.933*x3-43.0934)-1.508*x1.*(x6.^2+x7.^2)+7.477*(x6.^3+x7.^3)+0.7854*(x4.*x6.^2+x5.*x7.^2);
            f2=sqrt((745*x4./(x2.*x3)).^2+1.69e7)./(0.1*x6.^3);
            g1=-1./(x1.*x2.^2.*x3)+1/27; g2=-1./(x1.*x2.^2.*x3.^2)+1/397.5; g3=-x4.^3./(x2.*x3.*x6.^6)+1/1.93; g4=-x5.^3./(x2.*x3.*x7.^6)+1/1.93;
            g5=-x2.*x3+40;g6=-x1./x2+12;g7=-5+x1./x2;g8=-1.9+x4-1.5*x6;g9=-1.9+x5-1.1*x7;g10=-f2+1300;
            g11=-sqrt((745*x5./(x2.*x3)).^2+1.575e8)./(0.1*x7.^3)+1100;
            F=[f1 f2 obj.violation(g1,g2,g3,g4,g5,g6,g7,g8,g9,g10,g11)];
        end

        function F = re36(obj,x)
            x1=x(:,1);x2=x(:,2);x3=x(:,3);x4=x(:,4); f1=abs(6.931-(x3./x1).*(x4./x2)); f2=max(x,[],2); g1=0.5-f1/6.931;
            F=[f1 f2 obj.violation(g1)];
        end

        function F = re37(obj,x)
            a=x(:,1);h=x(:,2);o=x(:,3);t=x(:,4);
            f1=0.692+0.477*a-0.687*h-0.080*o-0.065*t-0.167*a.^2-0.0129*h.*a+0.0796*h.^2-0.0634*o.*a-0.0257*o.*h+0.0877*o.^2-0.0521*t.*a+0.00156*t.*h+0.00198*t.*o+0.0184*t.^2;
            f2=0.153-0.322*a+0.396*h+0.424*o+0.0226*t+0.175*a.^2+0.0185*h.*a-0.0701*h.^2-0.251*o.*a+0.179*o.*h+0.015*o.^2+0.0134*t.*a+0.0296*t.*h+0.0752*t.*o+0.0192*t.^2;
            f3=0.370-0.205*a+0.0307*h+0.108*o+1.019*t-0.135*a.^2+0.0141*h.*a+0.0998*h.^2+0.208*o.*a-0.0301*o.*h-0.226*o.^2+0.353*t.*a-0.0497*t.*o-0.423*t.^2+0.202*h.*a.^2-0.281*o.*a.^2-0.342*h.^2.*a-0.245*h.^2.*o+0.281*o.^2.*h-0.184*t.^2.*a-0.281*h.*a.*o;
            F=[f1 f2 f3];
        end

        function F = re41(obj,x)
            x1=x(:,1);x2=x(:,2);x3=x(:,3);x4=x(:,4);x5=x(:,5);x6=x(:,6);x7=x(:,7);
            f1=1.98+4.9*x1+6.67*x2+6.98*x3+4.01*x4+1.78*x5+1e-5*x6+2.73*x7; f2=4.72-0.5*x4-0.19*x2.*x3;
            vmbp=10.58-0.674*x1.*x2-0.67275*x2; vfd=16.45-0.489*x3.*x7-0.843*x5.*x6; f3=0.5*(vmbp+vfd);
            g1=1-(1.16-0.3717*x2.*x4-0.0092928*x3); g2=0.32-(0.261-0.0159*x1.*x2-0.06486*x1-0.019*x2.*x7+0.0144*x3.*x5+0.0154464*x6);
            g3=0.32-(0.214+0.00817*x5-0.045195*x1-0.0135168*x1+0.03099*x2.*x6-0.018*x2.*x7+0.007176*x3+0.023232*x3-0.00364*x5.*x6-0.018*x2.^2);
            g4=0.32-(0.74-0.61*x2-0.031296*x3-0.031872*x7+0.227*x2.^2); g5=32-(28.98+3.818*x3-4.2*x1.*x2+1.27296*x6-2.68065*x7);
            g6=32-(33.86+2.95*x3-5.057*x1.*x2-3.795*x2-3.4431*x7+1.45728);g7=32-(46.36-9.9*x2-4.4505*x1);g8=4-f2;g9=9.9-vmbp;g10=15.7-vfd;
            F=[f1 f2 f3 obj.violation(g1,g2,g3,g4,g5,g6,g7,g8,g9,g10)];
        end

        function F = re42(obj,x)
            L=x(:,1);B=x(:,2);D=x(:,3);T=x(:,4);Vk=x(:,5);CB=x(:,6); dispv=1.025*L.*B.*T.*CB; V=0.5144*Vk; g=9.8065; Fn=V./sqrt(g*L);
            aa=4977.06*CB.^2-8105.61*CB+4456.51;bb=-10847.2*CB.^2+12817*CB-6960.32; power=dispv.^(2/3).*Vk.^3./(aa+bb.*Fn);
            outfit=L.^0.8.*B.^0.6.*D.^0.3.*CB.^0.1; steel=0.034*L.^1.7.*B.^0.7.*D.^0.4.*CB.^0.5; mach=0.17*power.^0.9; light=steel+outfit+mach;
            ship=1.3*(2000*steel.^0.85+3500*outfit+2400*power.^0.8); cap=0.2*ship; DWT=dispv-light; running=40000*DWT.^0.3; sea=5000/24*Vk; daily=0.19*power*24/1000+0.2; fuel=1.05.*daily.*sea.*100; port=6.3*DWT.^0.8; cargo=DWT-daily.*(sea+5)-2*sqrt(DWT); portdays=2*(cargo/8000+0.5); rtpa=350./(sea+portdays); annual=cap+running+(fuel+port).*rtpa; annualCargo=cargo.*rtpa;
            f1=annual./annualCargo;f2=light;f3=-annualCargo; KB=0.53*T;BMT=((0.085*CB-0.002).*B.^2)./(T.*CB);KG=1+0.52*D;
            g1=L./B-6;g2=-L./D+15;g3=-L./T+19;g4=0.45*DWT.^0.31-T;g5=0.7*D+0.7-T;g6=500000-DWT;g7=DWT-3000;g8=0.32-Fn;g9=KB+BMT-KG-0.07*B;
            F=[f1 f2 f3 obj.violation(g1,g2,g3,g4,g5,g6,g7,g8,g9)];
        end

        function F = re61(obj,x)
            x1=x(:,1);x2=x(:,2);x3=x(:,3);f1=106780.37*(x2+x3)+61704.67;f2=3000*x1;f3=305700*2289*x2/(0.06*2289)^0.65;f4=250*2289*exp(-39.75*x2+9.9*x3+2.74);f5=25*(1.39./(x1.*x2)+4940*x3-80);
            g1=1-(0.00139./(x1.*x2)+4.94*x3-0.08);g2=1-(0.000306./(x1.*x2)+1.082*x3-0.0986);g3=50000-(12.307./(x1.*x2)+49408.24*x3+4051.02);g4=16000-(2.098./(x1.*x2)+8046.33*x3-696.71);g5=10000-(2.138./(x1.*x2)+7883.39*x3-705.04);g6=2000-(0.417*x1.*x2+1721.26*x3-136.54);g7=550-(0.164./(x1.*x2)+631.13*x3-54.48);
            F=[f1 f2 f3 f4 f5 obj.violation(g1,g2,g3,g4,g5,g6,g7)];
        end

        function F = re91(obj,x)
            % RE91 is stochastic in the source Python definition.  We use its
            % documented mean values to make PlatEMO runs reproducible.
            x1=x(:,1);x2=x(:,2);x3=x(:,3);x4=x(:,4);x5=x(:,5);x6=x(:,6);x7=x(:,7);x8=0.345;x9=0.192;x10=0;x11=0;
            f1=1.98+4.9*x1+6.67*x2+6.98*x3+4.01*x4+1.75*x5+1e-5*x6+2.73*x7;
            f2=max(0,1.16-0.3717*x2.*x4-0.00931*x2*x10-0.484*x3*x9+0.01343*x6*x10);
            f3=max(0,(0.261-0.0159*x1.*x2-0.188*x1*x8-0.019*x2.*x7+0.0144*x3.*x5+0.87570001*x5*x10+0.08045*x6*x9+0.00139*x8*x11+0.00001575*x10*x11)/0.32);
            f4=max(0,(0.214+0.00817*x5-0.131*x1*x8-0.0704*x1*x9+0.03099*x2.*x6-0.018*x2.*x7+0.0208*x3*x8+0.121*x3*x9-0.00364*x5.*x6+0.0007715*x5*x10-0.0005354*x6*x10+0.00121*x8*x11+0.00184*x9*x10-0.018*x2.^2)/0.32);
            f5=max(0,(0.74-0.61*x2-0.163*x3*x8+0.001232*x3*x10-0.166*x7*x9+0.227*x2.^2)/0.32);
            f6=max(0,((28.98+3.818*x3-4.2*x1.*x2+0.0207*x5*x10+6.63*x6*x9-7.77*x7*x8+0.32*x9*x10)+(33.86+2.95*x3+0.1792*x10-5.057*x1.*x2-11*x2*x8-0.0215*x5*x10-9.98*x7*x8+22*x8*x9)+(46.36-9.9*x2-12.9*x1*x8+0.1107*x3*x10))/3/32);
            f7=max(0,(4.72-0.5*x4-0.19*x2.*x3-0.0122*x4*x10+0.009325*x6*x10+0.000191*x11^2)/4);
            f8=max(0,(10.58-0.674*x1.*x2-1.95*x2*x8+0.02054*x3*x10-0.0198*x4*x10+0.028*x6*x10)/9.9);
            f9=max(0,(16.45-0.489*x3.*x7-0.843*x5.*x6+0.0432*x9*x10-0.0556*x9*x11-0.000786*x11^2)/15.7);
            F=[f1 f2 f3 f4 f5 f6 f7 f8 f9];
        end
    end
end

function y = safeDivide(a,b)
    % PlatEMO may provide a row or column batch depending on the caller.
    % Normalize both operands so logical indexing always has matching shape.
    a = a(:); b = b(:);
    y=zeros(size(a)); mask=(b~=0); y(mask)=a(mask)./b(mask);
end
