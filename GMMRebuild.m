%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Author(s): Tim J. Cooper
% Updated: 3/2/2018 (v.1.2)
% Gamma-mixture modelling (expectation-maximisation)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [alpha, beta, weight,logLH] = GMMRebuild(x,nC,maxIter,error_thresh,init_mode,init_alpha,init_beta,init_weight)

%% Data Normalisation & Binning
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
x = x(:)';
s_factor = sum(x)/length(x);
norm = x./s_factor;
[N,M] = hist(norm,linspace(min(norm(:)),max(norm(:)),150));
binned_data = N/(sum(N*(M(2)-M(1))));

%% Parameter & Distribution Initialisation (Pre-EM)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if strcmp(init_mode,'static') == 1
    alpha = init_alpha; beta = init_beta/s_factor; weight = init_weight;
elseif strcmp(init_mode,'kmeans') == 1
    idx = kmeans(x',nC,'Replicates',10);
    beta = zeros(1,nC); alpha = zeros(1,nC); weight = zeros(1,nC);
    for k=1:nC
        beta(1,k) = std(norm(idx==k))^2/mean(norm(idx==k));     %Method of Moments Estimation
        alpha(1,k) = (mean(norm(idx==k))/std(norm(idx==k)))^2;
        weight(1,k) = sum(idx==k)/sum(N);
    end
end
dist = zeros(nC,length(norm));
m_dist = zeros(1,length(norm));
for j=1:nC
    dist(j,:) = gampdf(norm,alpha(j),beta(j));
    m_dist = m_dist+dist(j,:).*weight(j);
end
for p=1:nC
    w(p,:) = dist(p,:).*weight(p)./m_dist;
end
alpha_trace{1} = alpha; beta_trace{1} = beta;

%% Expectation Maximisation (EM) Algorithm
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
error = inf; i=1; iter=i; c=0; options = []; options = statset(statset('gamfit'),options);
MLE = @(x_MLE,y_MLE) y_MLE-log(x_MLE)+psi(x_MLE); %Log-likelihood function
while (error>error_thresh && i<maxIter)
    m_dist = zeros(1,length(norm));
    c=c+1;
    for n=1:nC
        weight(n) = sum(w(n,:))/(sum(w(:)));
        A = log(sum(w(n,:).*norm)/sum(w(n,:)));
        B = sum(w(n,:).*log(norm+eps))/(sum(w(n,:))+eps);
        data_term = A-B;
        logLH(c,n) = MLE(alpha(n),data_term);
        if MLE(alpha(n),data_term) > 0
            upper = alpha(n); lower = upper/2;
            while MLE(lower,data_term) > 0
                upper = lower; lower = upper/2;
            end
        else
            lower = alpha(n); upper = lower*2;
            while MLE(upper,data_term) < 0
                lower = upper; upper = lower*2;
            end
        end
        boundaries = [lower upper];
        [ahat, ~, ~] = fzero(MLE,boundaries,options,data_term);
        alpha(1,n) = ahat;
        beta(n) = sum(w(n,:).*norm)/(sum(w(n,:))*alpha(n)+eps);
        dist(n,:) = gampdf(norm,alpha(n),beta(n));
        m_dist = m_dist+dist(n,:).*weight(n);
    end
    alpha_trace{i+1} = alpha;
    beta_trace{i+1} = beta;
    w = zeros(1,length(norm));
    for b=1:nC
        w(b,:) = dist(b,:).*weight(b)./m_dist;
        w(isnan(w)) = 1;
    end
    error = 0;
    for r=1:nC
        error = error+max(abs(weight - sum(w,2)'/(sum(w(:)))));
    end
    i=i+1;
    iter = [iter,i];
end
for t=1:nC
    weight(1,n) = sum(w(n,:))/(sum(w(:)));
end

%% Output
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fit_final = zeros(size(M));
for g=1:nC
    fit_final = fit_final+weight(g)*gampdf(M,alpha(g),beta(g));
    capture(:,g) = weight(g)*gampdf(M,alpha(g),beta(g));
end
beta = beta*s_factor;
[val,idx2] = sort(weight,'descend');
weight = val;
beta = beta(idx2);
alpha = alpha(idx2);