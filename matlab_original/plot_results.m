function plot_results()

%import stock prices of Berkshire Hathaway from locally stored file
%Header: Date,Open,High,Low,Close,Adj Close,Volume
data=readmatrix("C:/Users/Kai Koepchen#/Desktop/Universität zu Köln/Bachelorarbeit/Daten/BRK-B.csv");
open=data(:,2); close=data(:,5);returns=log(close./open)*100; 

%needed parameters for subsequent commands are defined
gibbs=5000;
n=length(returns);
alpha = 0.05;

%import the alpha0, alpha1, sigmav values over all 5000 iterations
alpha_0=readmatrix("alpha0_5k_BRK-Bcsv");
alpha_1=readmatrix("alpha1_BRK-B.csv");
sigma_v=readmatrix("sigma_v_5k_BRK-B.csv");

%mean and standard deviation of the parameter estimates are computed
alpha_0_std_deviation=sqrt(var(alpha_0(2500:5001)))
alpha0_std_error=alpha_0_std_deviation/sqrt(n)
alpha_1_std_deviation=sqrt(var(alpha_1(2500:5001)))
alpha1_std_error=alpha_1_std_deviation/sqrt(n)
sigma_v_std_deviation=sqrt(var(sigma_v(2500:5001)))
sigma_v_std_error=sigma_v_std_deviation/sqrt(n)

%Gibbs draws for all 1259 volatilites were stored in local csv
%conservativlely the first 50% of iterations are declared as burn in and
%the rest are used for inference by computing their arithmetic mean 
h_total = readmatrix("h_total_new_5k_.csv");
estimated_ln_ht=zeros(n,1);
for k=1:n
    sum_=0;
    for l=0.5*gibbs:gibbs+1
        sum_=sum_+log(h_total(k,l));
    end
    estimated_ln_ht(k)=sum_/(0.5*gibbs+1);
end


%using MATLAB command garch a model with 1 GARCH and 1 ARCH lag ist
%initiliaized. Using the returns parameters are estimated using maximum
%likelihood as defined in Section 4.1. Lastly, the condVar is inferred by
%using the model and its estimated parameters

model = garch("GARCHLags",1 , "ARCHLags",1) 
[estMdl, estParamCov, logL] = estimate(model, returns); 
condVar = infer(estMdl, returns); 
condVol=sqrt(condVar); 

%plot log returns
figure(1);
plot(returns); hold off; 
xlim([0, n]);
xlabel("Trading days","FontSize", 18,"FontName", "Times");
ylabel("r_t","FontSize", 18,"FontName", "Times")
legend("log returns", "Location","northwest","FontSize", 18,"FontName", "Times")
title("log returns of Berkshire Hathaway B","FontSize", 22, "FontName", "Times")

%plot sample ACF of rt, rt^2 and abs(rt) for 20 lags
figure(2);
sgtitle('Sample ACF of log returns and squarred log returns',"FontSize", 22, "FontName", "Times");
subplot(3,1,1);autocorr(returns);ylabel('Sample ACF of r_t',"FontSize", 18,"FontName", "Times"); 
subplot(3,1,2);autocorr(returns.^2);ylabel('Sample ACF of r^2_t',"FontSize", 18,"FontName", "Times");
subplot(3,1,3);autocorr(abs(returns));ylabel('Sample ACF of abs(r_t)',"FontSize", 18,"FontName", "Times");


%traceplot of alpha0,alpha1 and sigma_v
figure(3)
sgtitle('Trace Plots of Parameters',"FontSize", 22, "FontName", "Times");
subplot(3, 1, 1); plot(1:gibbs+1, alpha_0, '-'); xlabel('Iteration',"FontSize", 18,"FontName", "Times"); xlim([0, gibbs]);
ylabel('\alpha_0',"FontSize", 18,"FontName", "Times");
title('Trace Plot of \alpha_0',"FontSize", 18,"FontName", "Times");
subplot(3, 1, 2); plot(1:gibbs+1, alpha_1, '-'); xlabel('Iteration',"FontSize", 18,"FontName", "Times"); xlim([0, gibbs]);
ylabel('\alpha_1',"FontSize", 18,"FontName", "Times");
title('Trace Plot of \alpha_1',"FontSize", 18,"FontName", "Times");
subplot(3, 1, 3); plot(1:gibbs+1, sigma_v, '-'); xlabel('Iteration',"FontSize", 18,"FontName", "Times"); xlim([0, gibbs]);
ylabel('\sigma_v',"FontSize", 18,"FontName", "Times");
title('Trace Plot of \sigma_v',"FontSize", 18,"FontName", "Times");
xlim([0, gibbs]);

%plot rt and inferred volatility using GARCH(1,1) model
figure(4);
plot(returns); hold on; plot(condVol);hold off; 
xlim([0, n]);
xlabel("Trading days","FontSize", 18,"FontName", "Times");
ylabel("r_t, h_t","FontSize", 18,"FontName", "Times")
legend("log returns", "inferred volatilities of GARCH(1,1) model", "Location","northwest","FontSize", 18,"FontName", "Times")
title("log returns, inferred volatilities of GARCH(1,1) model","FontSize", 22, "FontName", "Times")


%plot rt and inferred volatility using SV model
figure(5);
plot(returns); hold on; plot(exp(estimated_ln_ht));hold off;
xlim([0, n]);
xlabel("Trading days","FontSize", 18,"FontName", "Times");
ylabel("r_t, h_t","FontSize", 18,"FontName", "Times")
legend("log returns", "inferred volatilities of SV model", "Location","northwest", "FontSize", 18,"FontName", "Times")
title("log returns, inferred volatilities of SV model", "FontSize", 22, "FontName", "Times")


end
