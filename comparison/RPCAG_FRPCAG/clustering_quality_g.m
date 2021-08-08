function [err_rpcag, err_frpcag] = clustering_quality_g(L_rpcag, L_frpcag, param_data)

%%
if ~isempty(L_rpcag)
 [ ~, cluster_idx_rpcag, ~,~,~ ] = kmeans_fast( L_rpcag,param_data.K,2,0);
 err_rpcag = 1 - clustering_purity(cluster_idx_rpcag, param_data.Labels);
else
    err_rpcag = NaN;
end

 if ~isempty(L_frpcag)
 [ ~, cluster_idx_frpcag, ~,~,~ ] = kmeans_fast( L_frpcag,param_data.K,2,0);
 err_frpcag = 1 - clustering_purity(cluster_idx_frpcag, param_data.Labels);
 else
     err_frpcag = NaN;
 end
end