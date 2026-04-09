function MCMC_estimation
%import data from local file
%Header: Date,Open,High,Low,Close,Adj Close,Volume    Length: 1259
data=readmatrix("C:/Users/Kai Koepchen#/Desktop/Universität zu Köln/Bachelorarbeit/Daten/BRK-B.csv");
open=data(:,2); close=data(:,5);r=log(close./open)*100;

%%%GARCH(1,1) model is used to initialize ln(ht´s) of Gibbs Sampler:
model = garch("GARCHLags",1 , "ARCHLags",1); %GARCH(1,1) model intialized
[estMdl, estParamCov, logL] = estimate(model, r); %parameters estimated through ML estimation
condVar = infer(estMdl, r); condVol=sqrt(condVar); %conditional Volatilites are inferred 

%test goodness of fit 
et=r./condVol;
sq_et=et.^2;
[h,pValue,stat,cValue] = lbqtest(sq_et);
%Ljung-Box test of squarred error terms of the GARCH(1,1) estimates 
%exhibit a p-value of of 0.9644
%indicating that the residuals are not linearly dependent

%Initialize ln(ht) with the condVol of the GARCH(1,1) model
T = length(r);
lnh = zeros(T,1);
for t=1:T
    lnh(t) = log(condVol(t));
end
lnh = lnh(1:T);
h_t = exp(lnh); 


%define number of gibbs iterations and starting values for parameters
gibbs = 5000; 
m=10;alpha0_start=0.5;alpha1_start=0.8;sigma_start=0.1;
% Setting a seed ensures reproducibility in the results
rand('seed',123);
randn('seed',123);

%parameter draws will be stored in vector of size gibbs+1. First value will
%be starting value of the Gibbs sampler
alpha0     = zeros(gibbs+1,1);
alpha0(1)=alpha0_start;

alpha1= zeros(gibbs+1,1);
alpha1(1)     = alpha1_start;

sigma_v= zeros(gibbs+1,1);
sigma_v(1) = sigma_start;


%all Gibbs draws of the volatilities will be stored in a matrix. 
h_total=zeros(T,gibbs+1);
h_total(:, 1)=h_t;

%hyperparameters of the prior of alpha is defined
alpha_cov_start=[0.25 0; 0 0.04];
alpha_mean_start=[alpha0_start alpha1_start];

%define grid an equally spaced grid for Griddy Gibbs with 3000 grid points
lower_grid = 0.001; upper_grid = 10; 
number_grid_points = 3000;  
grid_  = linspace(lower_grid, upper_grid, number_grid_points)';


% run Gibbs sampler for 5000 iterations
for ii=1:gibbs
    %First, the ht´s are drawn using a Griddy Gibbs sampler, because it
    %was only possible to obtain a unnormalized conditional posterior
    %distribution
    for t=2:T
          if t<T
            % for 1<t<n the Griddy Gibbs sampler samples from the 
            % distribution defined in (35):
            mu_t     = ( alpha0(ii)*(1-alpha1(ii))+alpha1(ii)*(log(h_t(t+1))+log(h_t(t-1))) )/(1+alpha1(ii)^2);
            sig_sq_t = (sigma_v(ii)^2)/(1+alpha1(ii)^2);
            %unnormalized evaluation of the conditional posterior
            %distribution of ht
            p      = (grid_.^(-1.5)).*exp(-(r(t)^2)./(2*grid_)-((log(grid_)-mu_t).^2)./(2*sig_sq_t)); 
          elseif t==T
            % for t=n the Griddy Gibbs sampler samples from the 
            % distribution defined in (36):
            p =(grid_.^(-1.5)).*exp((-r(t)^2./(2*grid_))-(log(grid_)-alpha0(ii)-alpha1(ii)*log(h_t(t-1)).^2/(2*sigma_v(ii))));
          end 
       % A less compuationally intensive way of gernerating random samples
       % is used here. Samples are drawn from the multinomial distribution.
       % This is equivalent to the generation of random samples through
       % inverse probability transform.
       % through p./sum(p) we obtain the normalized probabilities
       h_t(t)  = grid_(mnrnd(1,p./sum(p))==1);  

    end
    %results of the Griddy Gibbs Sampler are stored in h_total
    h_total(:, ii+1)=h_t;
    
    %now random samples from the conditional distribution of sigma_v is
    %drawn. This is the implementation of (41)
    vt=zeros(T,1);
    for k=3:T
        vt(k-2) = log(h_t(k)) - alpha0(ii) - alpha1(ii)*log(h_t(k-1));
    end
    sum_vt = sum(vt.^2);
    const = m * sigma_v(ii) + sum_vt;
    draw = chi2rnd(m + T+1 - 1);
    %random draw of the chi squarred distribution is scaled with (1/const)
    %in order for it to be a random draw for sigma_v
    sigma_new = draw * (1 / const);
    %current sigma_v sample is stored in the sigma_v vector
    sigma_v(ii+1)=sigma_new;

    
    %%Implementation of the alpha sampler as discussed in (39)
    zt = zeros(T,2);
    zt_square=zeros(2,2);
    zt_lnh=zeros(1,2);
    for l=3:T
        zt(l-2,:)=[1,log(h_t(l-1))];
    end
    for k = 1:length(zt)
    zt_square = zt_square + zt(k, :)' * zt(k, :);
    end
    for u =3:T
        zt_lnh=zt_lnh+zt(u-2,:)*log(h_t(u));
        
    end
    alpha_cov_working = inv((zt_square / sigma_v(ii)) + inv(alpha_cov_start));
    alpha_mean_working = alpha_cov_working * (transpose(zt_lnh)/ sigma_v(ii) + inv(alpha_cov_start) * transpose(alpha_mean_start));
    alpha_new= mvnrnd(alpha_mean_working', alpha_cov_working);
    %current alpha samples are stored in alpha0 and alpha1 vector
    alpha0(ii+1)=alpha_new(1);
    alpha1(ii+1)=alpha_new(2);
end

%results of Gibbs samplers are stored in a csv for future plots, analysis
%and computations
writematrix(h_total,'h_total_new.csv') 
writematrix(sigma_v,'sigma_v_new.csv') 
writematrix(alpha0,'alpha0_new.csv') 
writematrix(alpha1,'alpha1_new.csv') 

end

