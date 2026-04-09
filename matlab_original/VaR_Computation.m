function VaR_Computation

%import stock prices of Berkshire Hathaway from locally stored file
%Header: Date,Open,High,Low,Close,Adj Close,Volume
data=readmatrix("C:/Users/Kai Koepchen#/Desktop/Universität zu Köln/Bachelorarbeit/Daten/BRK-B.csv");
open=data(:,2); 
close=data(:,5);
%standard returns are not computed indivdually. (1) has already been inserted in(2) 
returns=log(close./open)*100; 

%needed parameters for subsequent commands are defined
gibbs=5000;
numDays = length(returns);
alpha = 0.05;

%Gibbs draws for all 1259 volatilites were stored in local csv
%conservativlely the first 50% of iterations are declared as burn in and
%the rest are used for inference by computing their arithmetic mean 
lnh_total = readmatrix("h_total_new_5k_.csv");
estimated_ln_ht=zeros(numDays,1);
for k=1:numDays
    sum_=0;
    for l=0.5*gibbs:gibbs+1
        sum_=sum_+log(lnh_total(k,l));
    end
    estimated_ln_ht(k)=sum_/(0.5*gibbs+1);
end


%using MATLAB command garch a model with 1 GARCH and 1 ARCH lag is
%initiliaized. Using the returns parameters are estimated using maximum
%likelihood as defined in Section 4.1. Lastly, the condVar is inferred by
%using the model and its estimated parameters
model = garch("GARCHLags",1 , "ARCHLags",1) 
[estMdl, estParamCov, logL] = estimate(model, returns); 
condVar = infer(estMdl, returns); 
condVol=sqrt(condVar); 


% Value at Risk estimates are computed for every day using GARCH(1,1) and
% SV model volatility estimates
VaR_estimate_GARCH = norminv(alpha, 0, condVol);
Var_estimate_SV = norminv(alpha, 0, exp(estimated_ln_ht));


%For visual inspection the returns, inferred volatilites and VaR estimates
%are plotted. Figure (1) denotes the estimates of the GARCH(1,1) model and
%Figure (2) the estimates of the SV model
figure(1);
plot(returns); hold on; plot(condVol);hold on; plot(VaR_estimate_GARCH);hold off;
xlim([0, numDays]);
xlabel("Trading days","FontSize", 18,"FontName", "Times");
ylabel("r_t, h_t,VAR_\alpha","FontSize", 18,"FontName", "Times")
legend("log returns", "inferred volatilities of GARCH(1,1) model","VaR estimtates of GARCH(1,1) model", "Location","northwest","FontSize", 18,"FontName", "Times")
title("log returns, inferred volatilities and VaR estimtates of GARCH(1,1) model","FontSize", 22, "FontName", "Times")


figure(2);
plot(returns); hold on; plot(exp(estimated_ln_ht));hold on; plot(Var_estimate_SV);hold off;
xlim([0, numDays]);
xlabel("Trading days","FontSize", 18,"FontName", "Times");
ylabel("r_t, h_t,VAR_\alpha","FontSize", 18,"FontName", "Times")
legend("log returns", "inferred volatilities of SV model","VaR estimtates of SV model", "Location","northwest", "FontSize", 18,"FontName", "Times")
title("log returns, inferred volatilities and VaR estimtates of SV model","FontSize", 22, "FontName", "Times")

%%% Coverage Tests for GARCH(1,1) model
% hit rate sequence for VaR estimates 
hit_rate_GARCH = zeros(numDays, 1);
for i = 1:numDays
    if returns(i) <= VaR_estimate_GARCH(i)
        hit_rate_GARCH(i) = 1;
    end
end
n1=sum(hit_rate_GARCH); 
n0=numDays-n1;
pof_ml_estimate=n1/numDays;

%First the Likelihood under the Null Hypothesis and then the likelihood
%under the alterantive hypothesis is computed using equation (45) and (46)
L0_uc_GARCH=(1-alpha)^(n0)*alpha^(n1)
L1_uc_GARCH=(1-pof_ml_estimate)^(n0)*pof_ml_estimate^(n1)

%The Likelihood ratio is computed using (47) and compared with the critical
%value arising from the property that the likelihood ratio is chi-squarred
%distributed
LR_uc_GARCH=-2*log(L0_uc_GARCH/L1_uc_GARCH)
critical_value=chi2inv(1-alpha,1);
%The GARCH(1,1) doesn´t fail the unconditional covergae test as it exhibits
%a LR_uc=0.2689 fails to surpass the critical value of 3.8415


%---------------------------------------------------------
%%idependence test
% Sequence to find transitions between the states i and j
tr = hit_rate_GARCH(2:end) - hit_rate_GARCH(1:end-1);
% Transitions: nij denotes state i is followed by state j nij times
n01 = sum(tr == 1);
n10 = sum(tr == -1);
n11 = sum(hit_rate_GARCH(tr == 0) == 1);
n00 = sum(hit_rate_GARCH(tr == 0) == 0);

% number of times in the state i or j
n0 = n01 + n00;
n1 = n10 + n11;
n = n0 + n1;

% Likelihood ratios are computed using the ML estimates as expressed in
% (49), (50), (51) and (50)
L1_ind_GARCH=(n00/(n00+n01))^(n00)*(n01/(n00+n01))^(n01)*(n10/(n10+n11))^(n10)*(n11/(n10+n11))^(n11);
pi_2=(n01+n11)/(n00+n10+n01+n11);
L2_ind_GARCH=(1-pi_2)^(n00+n10)*pi_2^(n01+n11);
%The Likelihood ratio is computed using (53) and compared with the critical
%value arising from the property that the likelihood ratio is chi-squarred
%distributed
LR_ind_GARCH=-2*log(L2_ind_GARCH/L1_ind_GARCH); 
critical_value=chi2inv(1-alpha,1);
%The GARCH(1,1) doesn´t fail the indepdence test as it exhibits
%a LR_uc=2.9586 fails to surpass the critical value of 3.8415


%---------------------------------------------------------


%%% Coverage Tests for SV model
% hit rate sequence for VaR estimates 
hit_rate_SV = zeros(numDays, 1);
for i = 1:numDays
    if returns(i) <= Var_estimate_SV(i)
        hit_rate_SV(i) = 1;
    end
end

%%unconditional coverage test
n1=sum(hit_rate_SV); 
n0=numDays-n1;
pof_ml_estimate_SV=n1/numDays;

%First the Likelihood under the Null Hypothesis and then the likelihood
%under the alterantive hypothesis is computed using equation (45) and (46)
L0_uc_SV=(1-alpha)^(n0)*alpha^(n1)
L1_uc_SV=(1-pof_ml_estimate_SV)^(n0)*pof_ml_estimate_SV^(n1)

%The Likelihood ratio is computed using (47) and compared with the critical
%value arising from the property that the likelihood ratio is chi-squarred
%distributed
LR_uc_SV=-2*log(L0_uc_SV/L1_uc_SV)
critical_value=chi2inv(1-alpha,1);
%The SV model doesn´t fail the unconditional covergae test as it exhibits
%a LR_uc=0.8375 fails to surpass the critical value of 3.8415


%---------------------------------------------------------
%%idependence test
% Sequence to find transitions between the states i and j
tr = hit_rate_SV(2:end) - hit_rate_SV(1:end-1);
% Transitions: nij denotes state i is followed by state j nij times
n01 = sum(tr == 1);
n10 = sum(tr == -1);
n11 = sum(hit_rate_SV(tr == 0) == 1);
n00 = sum(hit_rate_SV(tr == 0) == 0);

% number of times in the state i or j
n0 = n01 + n00;
n1 = n10 + n11;
n = n0 + n1;

% Likelihood ratios are computed using the ML estimates as expressed in
% (49), (50), (51) and (50)
L1_ind_SV=(n00/(n00+n01))^(n00)*(n01/(n00+n01))^(n01)*(n10/(n10+n11))^(n10)*(n11/(n10+n11))^(n11);
pi_2=(n01+n11)/(n00+n10+n01+n11);
L2_ind_SV=(1-pi_2)^(n00+n10)*pi_2^(n01+n11);
%The Likelihood ratio is computed using (53) and compared with the critical
%value arising from the property that the likelihood ratio is chi-squarred
%distributed
LR_ind_SV=-2*log(L2_ind_SV/L1_ind_SV); 
critical_value=chi2inv(1-alpha,1);
%The SV doesn´t fail the indepdence test as it exhibits
%a LR_uc=0.1138 fails to surpass the critical value of 3.8415

end
