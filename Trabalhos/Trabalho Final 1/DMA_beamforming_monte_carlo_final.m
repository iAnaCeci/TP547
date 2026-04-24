clear; clc; close all;

%% ================= CONFIG =================
f_0 = 5e9;
c0 = physconst('LightSpeed');
lambda = c0/f_0;

boresight_gain = 2;

Delta_EH = 500e-6;
Delta_U  = 50e-6;

x_min = 0; x_max = 10;
y_min = 0; y_max = 10;
z_max = 3;

cenarios = [1 3];

%% Grade para mapa de calor
gradex = x_min:0.2:x_max;
gradey = y_min:0.2:y_max;
Nxg = length(gradex);
Nyg = length(gradey);

%% Monte Carlo
MC = 100;

%% BA
t_max = 200;
n_bats = 100;

%% RESULTADOS
PTX = zeros(2,length(cenarios),MC);
SUCCESS = zeros(2,length(cenarios),MC);

% mapa médio: (modo, cenário, x, y, mc)
PRX_MAP = zeros(2, length(cenarios), Nxg, Nyg, MC);

%% LOOP PRINCIPAL
for MODE = 1:2   % 1=digital | 2=DMA
    
    for c = 1:length(cenarios)
        cenario = cenarios(c);
        
        for mc = 1:MC
            
            rng(mc);
            
            %% ================= USUARIOS =================
            if cenario == 1
                k_loc = [4 5 1];
            else
                k_loc = [4 5 1; 5 5 1];
            end
            
            %% ================= ANTENA =================
            L = 0.2;
            DMA_WG_dist = lambda/2;
            DMA_element_dist = lambda/5;
            
            RFC_num = floor(L/DMA_WG_dist);
            passive_num = floor(L/DMA_element_dist);
            
            N = RFC_num * passive_num;
            
            DMA_loc = [5 5 z_max];
            
            DMA_element_loc = zeros(N,3);
            for i = 1:RFC_num
                for j = 1:passive_num
                    idx = (i-1)*passive_num + j;
                    DMA_element_loc(idx,:) = [DMA_loc(1) + j*DMA_element_dist, ...
                                              DMA_loc(2) + i*DMA_WG_dist, ...
                                              DMA_loc(3)];
                end
            end
            
            %% ================= CANAL DOS USUARIOS =================
            kk_loc = [k_loc(:,2) k_loc(:,1) k_loc(:,3)];
            channel_vec = Do_Channels(DMA_element_loc, kk_loc, boresight_gain, lambda);
            
            %% ================= MATRIZ H (DMA) =================
            H = zeros(N,N);
            for i = 1:RFC_num
                for l = 1:passive_num
                    idx = (i-1)*passive_num + l;
                    H(idx,idx) = H_DMA(f_0,L,(l-1)*DMA_element_dist);
                end
            end
            
            %% ================= BA INIT =================
            SOL = randn(N,1,n_bats) + 1i*randn(N,1,n_bats);
            erro = zeros(1,n_bats);
            
            for i = 1:n_bats
                if MODE == 1
                    erro(i) = sum(abs(SOL(:,:,i)).^2);
                else
                    erro(i) = sum(abs(H*SOL(:,:,i)).^2);
                end
            end
            
            [best_val,idx] = min(erro);
            best = SOL(:,:,idx);
            
            %% ================= BAT ALGORITHM =================
            for t = 1:t_max
                for b = 1:n_bats
                    
                    w_new = SOL(:,:,b) ...
                          + 0.1*(best - SOL(:,:,b)) ...
                          + 0.05*(randn(N,1) + 1i*randn(N,1));
                    
                    %% ===== DIGITAL =====
                    if MODE == 1
                        Ptx = sum(abs(w_new).^2);
                        P1  = abs(channel_vec(:,1)'*w_new).^2;
                        
                        if cenario == 1
                            ok = (P1 >= Delta_EH);
                        else
                            P2 = abs(channel_vec(:,2)'*w_new).^2;
                            ok = (P1 >= Delta_EH) && (P2 <= Delta_U);
                        end
                        
                    %% ===== DMA =====
                    else
                        Ptx = sum(abs(H*w_new).^2);
                        P1  = abs(channel_vec(:,1)'*H*w_new).^2;
                        
                        if cenario == 1
                            ok = (P1 >= Delta_EH);
                        else
                            P2 = abs(channel_vec(:,2)'*H*w_new).^2;
                            ok = (P1 >= Delta_EH) && (P2 <= Delta_U);
                        end
                    end
                    
                    if ok && Ptx < erro(b)
                        SOL(:,:,b) = w_new;
                        erro(b) = Ptx;
                        
                        if Ptx < best_val
                            best = w_new;
                            best_val = Ptx;
                        end
                    end
                end
            end
            
            %% ================= RESULTADOS =================
            PTX(MODE,c,mc) = best_val;
            
            if MODE == 1
                final_vec = best;
            else
                final_vec = H*best;
            end
            
            if cenario == 1
                SUCCESS(MODE,c,mc) = ...
                    (abs(channel_vec(:,1)'*final_vec).^2 >= Delta_EH);
            else
                SUCCESS(MODE,c,mc) = ...
                    (abs(channel_vec(:,1)'*final_vec).^2 >= Delta_EH) && ...
                    (abs(channel_vec(:,2)'*final_vec).^2 <= Delta_U);
            end
            
            %% ================= MAPA DE CALOR =================
            PRX = zeros(Nxg,Nyg);
            
            for ix = 1:Nxg
                for iy = 1:Nyg
                    % usa z=1 para ficar no plano dos usuários
                    test_point = [gradey(iy), gradex(ix), 1];
                    ctest = Do_Channels(DMA_element_loc, test_point, boresight_gain, lambda);
                    
                    if MODE == 1
                        PRX(ix,iy) = abs(ctest' * final_vec)^2;
                    else
                        PRX(ix,iy) = abs(ctest' * final_vec)^2;
                    end
                end
            end
            
            % normalização em dB
            PRX = PRX ./ max(PRX(:));
            PRX = 10*log10(PRX);
            PRX(PRX < -25) = -25;
            
            PRX_MAP(MODE,c,:,:,mc) = PRX;
            
            fprintf("MODE %d | Cen %d | MC %d\n", MODE, cenario, mc);
        end
    end
end

%% ================= RESULTADOS =================
fprintf('\n====== RESULTADOS ======\n');

for MODE = 1:2
    if MODE == 1
        disp('--- DIGITAL ---')
    else
        disp('--- DMA ---')
    end
    
    for c = 1:length(cenarios)
        fprintf('Cenario %d:\n', cenarios(c));
        fprintf('PTX medio = %.2e\n', mean(PTX(MODE,c,:)));
        fprintf('Sucesso = %.2f %%\n', 100*mean(SUCCESS(MODE,c,:)));
    end
end

%% ================= GRAFICO DE COMPARAÇÃO =================
bar_data = [mean(PTX(1,1,:)), mean(PTX(2,1,:));   % C1
            mean(PTX(1,2,:)), mean(PTX(2,2,:))];  % C3

figure;
bar(bar_data)
legend('Digital','DMA')
xticklabels({'Cenario 1','Cenario 3'})
ylabel('PTX')
title('Comparacao PTX')

%% ================= MAPAS DE CALOR MÉDIOS =================
for MODE = 1:2
    for c = 1:length(cenarios)
        
        PRX_mean = squeeze(mean(PRX_MAP(MODE,c,:,:,:), 5));
        
        figure('Color','w');
        contourf(gradex, gradey, PRX_mean', 12, 'LineWidth', 1);
        colorbar;
        colormap(jet);
        xlabel('X');
        ylabel('Y');
        
        if MODE == 1
            metodo = 'Beamforming Digital';
        else
            metodo = 'DMA';
        end
        
        title(sprintf('%s - Mapa de calor medio - Cenario %d', metodo, cenarios(c)));
        hold on;
        
        if cenarios(c) == 1
            k_loc = [4 5 1];
        else
            k_loc = [4 5 1; 5 5 1];
        end
        
        plot(k_loc(:,1), k_loc(:,2), 'ko', 'MarkerSize', 10, 'LineWidth', 2);
        
        for u = 1:size(k_loc,1)
            text(k_loc(u,1)+0.15, k_loc(u,2), sprintf('User %d',u), ...
                'Color','k','FontWeight','bold');
        end
    end
end

%% ================= FUNÇÕES =================
function h = H_DMA(f_0,L,x)
    c0 = physconst('LightSpeed');
    lambda = c0/f_0;
    alpha = 0.5;
    beta = 2*pi/lambda;
    h = exp(-x*(alpha + 1i*beta));
end

function channel_vec = Do_Channels(Y,X,boresight_gain,lambda)
    N = size(Y,1);
    M = size(X,1);
    channel_vec = zeros(N,M);

    for i = 1:N
        for m = 1:M
            d = norm(Y(i,:) - X(m,:));
            dz = abs(Y(i,3) - X(m,3));
            gain = (dz/d)^boresight_gain;
            channel_vec(i,m) = gain*(lambda/(4*pi*d))*exp(-1i*2*pi*d/lambda);
        end
    end
end