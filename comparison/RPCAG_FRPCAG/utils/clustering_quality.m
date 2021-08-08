function [err_rpcag, err_frpcag, err_pcagtv] = clustering_quality(L_rpcag, L_frpcag, L_pcagtv, param_data)

%%
if ~isempty(L_rpcag)
 [ ~, cluster_idx_rpcag, ~,~,~ ] = kmeans_fast( L_rpcag,param_data.K,2,0);
 err_rpcag = 1 - clustering_purity(cluster_idx_rpcag, param_data.Labels);
else
    err_rpcag = NaN;
end
 
 %
 if ~isempty(L_frpcag)
 [ ~, cluster_idx_frpcag, ~,~,~ ] = kmeans_fast( L_frpcag,param_data.K,2,0);
 err_frpcag = 1 - clustering_purity(cluster_idx_frpcag, param_data.Labels);
 else
     err_frpcag = NaN;
 end
 
 if ~isempty(L_pcagtv)
 [ ~, cluster_idx_pcagtv, ~,~,~ ] = kmeans_fast( L_pcagtv,param_data.K,2,0);
 err_pcagtv = 1 - clustering_purity(cluster_idx_pcagtv, param_data.Labels);
 else
     err_pcagtv = NaN;
 end
    
end