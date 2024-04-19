function [est,eval] = run_MOT_func_03(model,meas,SPENT,SANT,options)
    %================================ Beschreibung =======================================================
    % Implementierung eines Multi Object Trackers (MOT) mit den
    % folgenden Algorithmen für die jeweilige Teilaufgabe.
    % - Zustandsschätzung (Prediction): 
    %       a. Kalman Filter (KF)
    %       b. LSTM Network - SPENT: single prediction network trained
    % - Assoziation der Objekte (Assosiation):
    %       a. Euklidische-Distanzmatrix / Hungarian Algo. / Global Nearest Neighbor (GNN)
    %       b. SANT: single association network trained
    %=====================================================================================================
    
    


    %---------------------------------Setup/Init.--------------------------------------
    % Programmschalter
    diag_outprint_flag = 0;                      % InfoAusgabe in der Komandozeile
    eval_flag = 0;                               % Zusätzlicher Struct mit Analysedaten ausgeben, 1 = Ausgabe erzeugen

    % Ausgabe der Analysedaten
    eval.used_dz = "INIT";                                          % -- verwendete Distanzvarianz --
    eval.dz_kalman = cell(meas.K,1);                                % Distanzvarianz: Abweichung Messungen zu Kalmanprädiktion
    eval.dz_SPENT = cell(meas.K,1);                                 % Distanzvarianz: Abweichung Messungen zu Netzwerkprädiktion
    eval.used_method_distmeas = "INIT";                             % -- verwendete Methode zur Berechnung Assoziationsmatrix_00 --
    eval.distmeas_mat_00 = cell(meas.K,1);                          % Distanzvarianzmatrix_00
    eval.used_input_SANT = "INIT";                                  % -- verwendete Tracks als Input für Assoziationsnetzwerk --
    eval.distmeas_mat_SANT = cell(meas.K,1);                        % Ergebnis des Netzwerks: Distanzvarianzmatrix_SANT (Berechnet aus class_prob_SANT)
    eval.association_SANT_clasAndUpdate = cell(meas.K,1);           % Ergebnis des Netzwerks: Zuordnung Messung zu Trackliste
    eval.used_assignment_mat = "INIT";                              % -- verwendete Assoziationsmatrix --
    eval.assignment_mat_00 = cell(meas.K,1);                        % Ergebnis Hungarian Algorithmus mit Input distmeas_mat_00
    eval.assignment_mat_SANT = cell(meas.K,1);                      % Ergebnis Hungarian Algorithmus mit Input distmeas_mat_SANT

    

    % ID Tag / Kennzeichnung
    newIDTag = 1;
    
    % Ausgabevariablen
    est.X_kalman = cell(meas.K,1);                                              % Zustandsschätzwerte Kalman Filter
    est.X_SPENT = cell(meas.K,1);                                               % Zustandsschätzwerte LSTM basierttes Netzwerk
    est.Z_associated = cell(meas.K,1);                                          % Assoziierte Beobachtungen bzw. Messwerte
    est.track_list = cell(meas.K,1);                                            % Erzeugte Tracks
    est.N = zeros(meas.K,1);
    est.L = cell(meas.K,1);
    est.Z_associated_SANT = cell(meas.K,1);                                     % Assoziierte Beobachtungen bzw. Messwerte (SANT)

    %---- Kalman Filter ----
    % Parameter
    filter.P_G= 0.01;                                                           % Größe des Gates in Prozent
    filter.gamma= chi2inv(filter.P_G,model.z_dim);                              % gamma = chi2inv(P_G,nu); gibt die inverse kumulative Verteilungsfunktion (icdf) der Chi-Quadrat-Verteilung mit Freiheitsgraden nu zurück, ausgewertet für die Wahrscheinlichkeitswerte in P_G.
    filter.gate_flag = 0;                                    % Gating on/off := 1/0
    % filter.L_max = 100;                                                         % Begrenzung der Anzahl der Kalman Filter Objekte (KFO)
    % filter.elim_threshold= 0.1;                                                 % Schwellwert
    filter.run_flag= 'disp';                                      % 'disp' oder 'silence' für die Ausgabe im laufenden Betrieb
    est.filter = filter;
    % Initalisierung
    p_e_update(1)= eps;
    p_ne_update(1)= 1-p_e_update(1);
    x_update_KALMAN(:,1)= [0.1;0;0.1;0;0.1];                                    % Zustände nach Aktualisierung durch eine neue Beobachtung (Init. A-priori-Schätzung)
    P_update(:,:,1)= diag([1 1 1 1 1]).^2;                                      % Kovarianzmatrix der Zustände (Init. mit Einheitsmatrix)
    tl_update(1) = [newIDTag];                                                  % Tag Liste (ID pro Track)
    
    %---- SPENT: single prediction network trained ----
    % Parameter 
    sig_SPENT_SANT = [9.12118102600540; 0.413175639733112; 16.8833362839276];         % Sandardardabweichung für x,y,z - Normierungswerte vom Datensatz Training
    mu_SPENT_SANT = [-1.264201163874999; 1.640543043595974; 25.4977462833363];        % Erwartungswert für x,y,z - Normierungswerte vom Datensatz Training
    % Initalisierung
    netList_SPENT{1} = resetState(SPENT);                                             % Hiddenstates pro Sequenz zurücksetzen;
    x_update_SPENT(:,1)= [0.1;0.1;0.1];

    %--- SANT
    SANT = resetState(SANT);                                                % States pro Sequenz zurücksetzen;

    %---- Datenassoziation Initalisierung ----
    m_corr_update(:,1)= [0.1;0;0.1;0;0.1];
    


    %--------------------------------- Rekursives Filter Verfahren ----------------------------------
    for k = 1:meas.K                                                                                  % k, aktueller Frame / Zeitschritt
                                                                                                      % state vector definition [X_kitti X_vel_kitti Z_kitti Z_vel_kitti Y_kitti]
        %----------------- Prädiktion (Prediction) -----------------
        [x_predict_KALMAN,P_predict] = kalmanFilter_predict_all(model,x_update_KALMAN,P_update);      % lokale Kalman Funktion aufrufen
        prob_ex_predict_KALMAN = model.P_S * p_e_update + model.p_b * p_ne_update;                    % Existenzwahrscheinlichkeit (Probability of existence)
        p_ne_predict = (1 - model.P_S) * p_e_update + (1 - model.p_b) * p_ne_update;                  % non-existence prob
        tl_predict = tl_update;                                                                       % ID tag prediction --> stays the same                                                                          
        outsiders = [];                                                                               % Messungen ohne Zuordnung in jedem Zeitschritt leer aufsetzen
    
        % Gating
        if filter.gate_flag
            [meas.Z{k}, outsiders] = gate_meas_gms(meas.Z{k},filter.gamma,model,x_predict_KALMAN,P_predict);     
        end
        %-----------------
            
        %----------------- Datenassoziation -----------------
        % ungarische Zuordnung: Distanzmatrix erzeugen
        m = size(meas.Z{k},2);                  % Anzahl der Messungen
        n = size(x_predict_KALMAN,2);           % Anzahl der Tracks
        distmeas_mat_00 = zeros(n,m);           % Distanzmaß Matrix Null initalisieren
        for i = 1:n
            for j = 1:m
                S_mat  = model.R + model.H * P_predict(:,:,i) * model.H';           % Innovationskovarianzmatrix
                z_hat = model.H * x_predict_KALMAN(:,i);                            % Innovationen 
                x_update_SPENT_phy = x_update_SPENT(:,i) .* sig_SPENT_SANT + mu_SPENT_SANT;   % SPENT: Umrechnung physikalische Werte und Übergabe
                dz_KALMAN = meas.Z{k}(:,j) - z_hat;                                 % Delta / Distanzvarianz: Abweichung Messungen zu Prädiktion
                dz_SPENT = meas.Z{k}(:,j) - x_update_SPENT_phy([1 3 2]);            % SPENT: Delta / Distanzvarianz: Abweichung Messungen zu Prädiktion, % meas.Z kommt mit x / z / y rein
                
                % Analyse: Ausgabe von zusätzlichen Werten
                if eval_flag == 1
                    eval.dz_kalman{k} = [eval.dz_kalman{k} dz_KALMAN];   
                    eval.dz_SPENT{k} = [eval.dz_SPENT{k} dz_SPENT];      
                end

                %----- Berechnung der Distanzmatrix -----
                % Auswahl der Distanzvarianz
                eval.used_dz = options.dz;
                switch eval.used_dz
                    case "dz_KALMAN"
                        dz = dz_KALMAN;
                    case "dz_SPENT"
                        dz = dz_SPENT;
                end
                % Distanzmaß
                mahalanobisDist = sqrt(dz'*inv(S_mat)*dz);                              % V1: Mahalanobis-Distanz
                euclidDist = sqrt(dz'*dz);                                              % V2: Euklidische-Distanz
                % V3: baderholzscheDist
%                z_ki_SPENT = [meas.Z{k}(1,j);meas.Z{k}(3,j);meas.Z{k}(2,j)];            % Auswahl Zustandswerte: X,Y,Z
%                z_ki_SPENT_norm = (z_ki_SPENT - mu_SPENT_SANT) ./ sig_SPENT_SANT;       % Normierte Daten als Netzwerkinput
%                [SPENT_tmpNet,~] = predictAndUpdateState(netList_SPENT{i},z_ki_SPENT_norm);  
%                baderholzscheDist = sqrt((netList_SPENT{i}.Layers(2, 1).HiddenState - SPENT_tmpNet.Layers(2, 1).HiddenState)'*(netList_SPENT{i}.Layers(2, 1).HiddenState - SPENT_tmpNet.Layers(2, 1).HiddenState));

                % Auswahl Distanzmaß
                eval.used_method_distmeas = options.distmeas;
                switch eval.used_method_distmeas
                    case "euclidDist"
                        distmeas_mat_00(i,j) = euclidDist;
                    case "mahalanobisDist"
                        distmeas_mat_00(i,j) = mahalanobisDist;
%                    case "baderholzscheDist"
%                        distmeas_mat_00(i,j) = baderholzscheDist;
                end              
                % Distanzwerte > definierter Grenzwert := 9999
                if (distmeas_mat_00(i,j) > model.maxAssociationThres)
                    distmeas_mat_00(i,j) = 9999;
                end
                %----- 
            end
        end
        % ungarische Zuordnung: Hungarian Algorithmus ausführen
        [assignment_mat_00,~]= Hungarian(distmeas_mat_00);                            % Zuordnungsmatrix: Spalte = Messungen, Reihen = Tracks
        % Analyse: Ausgabe von zusätzlichen Werten
        if eval_flag == 1
            eval.distmeas_mat_00{k} = distmeas_mat_00;
            eval.assignment_mat_00{k} = assignment_mat_00;
        end

        % Datenassoziation mit SANT: single association network trained
        distmeas_mat_SANT = zeros(size(distmeas_mat_00));
        for j = 1:m
            input_SANT = zeros(6,size(x_update_SPENT,2));
            
            z_ki_SANT = [meas.Z{k}(1,j);meas.Z{k}(3,j);meas.Z{k}(2,j)];                % Auswahl Zustandswerte: X,Y,Z
            input_SANT(4:6,1) = (z_ki_SANT - mu_SPENT_SANT) ./ sig_SPENT_SANT;         % Normierte Daten als Netzwerkinput

            % Auswahl der Tracks / des Updateschritts
            eval.used_input_SANT = options.input_SANT;
            switch eval.used_input_SANT
                case "x_update_SPENT"
                    if size(x_update_KALMAN,2) ~= 0             % wenn Update stattgefunden hat (Tracks vorhanden sind)
                        input_SANT(1:3,:) = x_update_SPENT;
                    end
                case "x_update_KALMAN"
                    if size(x_update_KALMAN,2) ~= 0             % wenn Update stattgefunden hat (Tracks vorhanden sind)
                        input_SANT(1:3,:) = x_update_KALMAN([1 3 5],:);
                    end
            end
            [SANT,eval.association_SANT_clasAndUpdate{k}(j),class_prob_SANT] = classifyAndUpdateState(SANT,input_SANT);     % Netzwerkklassifikation durchführen
            distmeas_mat_SANT(1:min(n,m),j) = (1-class_prob_SANT(1:min(n,m))');                                             % Abhängig von der Anzahl der Tracks und Messungen
        end
        [assignment_mat_SANT,~]= Hungarian(distmeas_mat_SANT); 

        % Analyse: Ausgabe von zusätzlichen Werten
        if eval_flag == 1
            eval.distmeas_mat_SANT{k} = distmeas_mat_SANT;
            eval.assignment_mat_SANT{k} = assignment_mat_SANT;
        end
        %-----------------

        % Auswahl der Assoziationsmatrix
        eval.used_assignment_mat = options.assignment_mat;
        switch eval.used_assignment_mat
            case "assignment_mat_00"
                assignment_mat = assignment_mat_00;
            case"assignment_mat_SANT"
                assignment_mat = assignment_mat_SANT;
        end

        %----------------- Aktualisierung / Korrektur (Update) -----------------
        x_update_KALMAN = x_predict_KALMAN;                         % Übergabe der Zustandsprädiktion für den Update-Schritt
        m_corr_update = x_predict_KALMAN;                           % Übergabe der Zustandsprädiktion für die Zuweisung neuer und assoziierter Beobachtungen
        P_update = P_predict;                           
        p_e_update = prob_ex_predict_KALMAN;
        p_ne_update = p_ne_predict;
        tl_update = tl_predict;
        update_SO_flag = zeros(size(x_update_KALMAN,2),1);
        for ii = 1:size(assignment_mat,2)                                       % ii für für Anzahl der Sensorobjekte im aktuellen Frame
            asso_idx = find(assignment_mat(:,ii) == 1);                         % Index assoziiertes SO mit Track
            if distmeas_mat_00(asso_idx,ii) < model.maxAssociationThres         % Grenzwert in Meter
                % Update, Betrachtung 01: assoziiertes Sensorobjekt (SO) vorhanden 
                % Kalman Update durchführen
                [qz_temp,x_out_temp,P_temp] = kalman_update_single(meas.Z{k}(:,ii),model.H,model.R,x_update_KALMAN(:,asso_idx),P_predict(:,:,asso_idx));
                x_update_KALMAN(:,asso_idx) = x_out_temp;
                z_ki = [meas.Z{k}(1,ii); 0; meas.Z{k}(2,ii); 0; meas.Z{k}(3,ii)];                                 % Neue Beobachtung i zum aktuellen Zeitschritt k
                m_corr_update(:,asso_idx) = z_ki;                                           
                % SPENT Update durchführen
                z_ki_SPENT = [meas.Z{k}(1,ii);meas.Z{k}(3,ii);meas.Z{k}(2,ii)];                                   % Auswahl Zustandswerte: X,Y,Z
                z_ki_SPENT_norm = (z_ki_SPENT - mu_SPENT_SANT) ./ sig_SPENT_SANT;                                 % Normierte Daten als Netzwerkinput
                [netList_SPENT{asso_idx},x_SPENT_temp] = predictAndUpdateState(netList_SPENT{asso_idx},z_ki_SPENT_norm);   
                x_update_SPENT(:,asso_idx) = x_SPENT_temp;
                % Update Flag setzen, Sensorobjekt vorhanden und für Update genutzt
                update_SO_flag(asso_idx) = 1;
                % Normalisierung im Updateschritt 
                p_e_norm = 1 / (prob_ex_predict_KALMAN(asso_idx) * model.P_D + p_ne_predict(asso_idx) * model.p_c);
                p_e_update(asso_idx) = p_e_norm * model.P_D * prob_ex_predict_KALMAN(asso_idx);
                p_ne_update(asso_idx) = p_e_norm * model.p_c * p_ne_predict(asso_idx);
            else
                % Update, Betrachtung 00: Initalisierung Tracks, erstes Aufkommen eines Sensorobjekts
                outsiders = [outsiders meas.Z{k}(:,ii)];     % Messungen ohne Zuordnung einsammeln (outsiders)
                for jj = 1:size(outsiders,2)
                    x_out_temp = [outsiders(1,jj); 0; outsiders(2,jj); 0;outsiders(3,jj)];

                    P_temp = model.B_birth;
                    p_e_temp = model.p_b;

                    x_update_KALMAN = cat(2,x_update_KALMAN,x_out_temp);
                    m_corr_update = cat(2,m_corr_update,x_out_temp);

                    P_update = cat(3,P_update,P_temp);
                    p_e_update = cat(1,p_e_update, p_e_temp);
                    p_ne_update = cat(1,p_ne_update, 1-p_e_temp);
                    tl_update = cat(1, tl_update, newIDTag);                            % neue IDTag zuweisen
                    newIDTag = newIDTag + 1;                                            % IDTag inkrementieren

                    % SPENT: nicht zugeordnetes Sensorobjekt normieren und den Prädiktionen hinzufügen
                    m_temp_Meas = [outsiders(1,jj); outsiders(3,jj); outsiders(2,jj)];
                    m_temp_Meas_norm = (m_temp_Meas - mu_SPENT_SANT) ./ sig_SPENT_SANT;
                    x_update_SPENT = cat(2,x_update_SPENT,m_temp_Meas_norm);

                    netList_SPENT{end+1} = resetState(SPENT);
                end
            end
        end
        % Update, Betrachtung 02: Sensorobjekt (SO) für bestehenden Track nicht vorhanden (missed detections)
        for ii = 1:length(update_SO_flag)
            if update_SO_flag(ii) ~= 1     % Prüfe, 0 --> Sensorobjekt fehlt
                p_e_norm = 1 / (prob_ex_predict_KALMAN(ii)*(1-model.P_D) + p_ne_predict(ii)*(1-model.p_c));
                p_e_update(ii) = p_e_norm *(1-model.P_D)* prob_ex_predict_KALMAN(ii);
                p_ne_update(ii) = p_e_norm * (1-model.p_c)* p_ne_predict(ii);
                %SPENT predict
                [netList_SPENT{ii},x_SPENT_temp] = predictAndUpdateState(netList_SPENT{ii},x_update_SPENT(:,ii));
                x_update_SPENT(:,ii) = x_SPENT_temp;
                m_corr_update(:,ii) = [0;0;0;0;0]; % Fallbehandlung: missed detections
            end
        end

        %----------------- Track Managment -----------------
        %--- Delete, schwellwertabhängig
        m_corr_update = m_corr_update(:,p_e_update > model.delete_thr);     % Messungen ohne Zuordnung 
        % SPENT
        x_update_SPENT = x_update_SPENT(:,p_e_update > model.delete_thr);   
        netList_SPENT = netList_SPENT(1,p_e_update > model.delete_thr);     
        % KALMAN
        x_update_KALMAN = x_update_KALMAN(:,p_e_update > model.delete_thr); 
        P_update = P_update(:,:,p_e_update > model.delete_thr);
        p_ne_update = p_ne_update(p_e_update > model.delete_thr);
        tl_update = tl_update(p_e_update > model.delete_thr);
        p_e_update = p_e_update(p_e_update > model.delete_thr);
    
        %--- Track Bestätigung, schwellwertabhängig
        idx = find(p_e_update > model.confirmed_trck);
        for j=1:length(idx)
            % KALMAN
            est.X_kalman{k}= [est.X_kalman{k} x_update_KALMAN(:,idx(j))];
            % SPENT: Umrechnung physikalische Werte und Übergabe
            x_update_SPENT_phy = x_update_SPENT(:,idx(j)) .* sig_SPENT_SANT + mu_SPENT_SANT;
            tmp = [x_update_SPENT_phy(1); 0 ; x_update_SPENT_phy(3); 0; x_update_SPENT_phy(2)];
            est.X_SPENT{k}= [ est.X_SPENT{k} tmp];
            est.Z_associated{k} = [ est.Z_associated{k} m_corr_update(:,idx(j))];
            est.N(k)= est.N(k)+1;
            est.L{k}= [];
            est.track_list{k} = [est.track_list{k} tl_update(idx(j))];
        end
        
        if diag_outprint_flag == 1
            %--- Ausgabe Kalman-Werte
            if ~strcmp(filter.run_flag,'silence')
                disp([' time = ',num2str(k),...
                     ' #est mean = ' num2str(sum(p_e_update),4),...
                     ' #est card = ' num2str(est.N(k),4),...
                     ' #gaus orig = ',num2str(length(p_e_update)) ]);
            end
        end
    end
    %---Ausgabe Ende
    disp('MOT FIN');

    %tracks need to have IDs aufsteigend and need to be reordered
    uniqueIDs = unique([est.track_list{:}],'first');
    est.total_tracks = length(uniqueIDs);
    for k=1:meas.K
        [~,newUniqueID_inds] = ismember(est.track_list{k}, uniqueIDs);
        est.track_list{k} = newUniqueID_inds;
    end
end
%% 
% Lokale Hilfsfunktionen
% 
% %--------------------------------- Kalman Filter: Prädiktion (Prediction) 
% ----------------------------------

function [m_predict,P_predict] = kalmanFilter_predict_all(model,x_update_past,P_update_past)      

    plength= size(x_update_past,2);
    
    m_predict = zeros(size(x_update_past));
    P_predict = zeros(size(P_update_past));
    
    for idxp=1:plength
        [m_temp,P_temp] = kalman_predict_one(model.F,model.Q,x_update_past(:,idxp),P_update_past(:,:,idxp));
        m_predict(:,idxp) = m_temp;
        P_predict(:,:,idxp) = P_temp;
    end
end

function [m_predict,P_predict] = kalman_predict_one(F,Q,m,p_update_temp)    % Q: Systemrauschen bzw. Prozessrauschen (W,V auf zeitkontinuierliche Erklärung abstahiert)
    m_predict = F*m;                                                        % F: Übergangsmatrix (Transitionmatix) 
    P_predict = Q + F * p_update_temp * F';                                 % P: Kovarianzmatrix (gekoppelte Zustände)
    % BEI KEINEM UPDATE STEIGT DIE UNSICHERHEIT AUF GRUND DER SUMMIERUNG DER KOVARIANZMATRIX
end
%% 
% %--------------------------------- Kalman Filter: Korrektur (Update) ----------------------------------

function [qz_temp,m_temp,P_temp] = kalman_update_single(z,H,R,m,P)

    mu = H*m;
    S  = R+H*P*H'; Vs= chol(S); det_S= prod(diag(Vs))^2; inv_sqrt_S= inv(Vs); iS= inv_sqrt_S*inv_sqrt_S';
    K  = P*H'*iS;
    
    qz_temp = exp(-0.5*size(z,1)*log(2*pi) - 0.5*log(det_S) - 0.5*dot(z-repmat(mu,[1 size(z,2)]),iS*(z-repmat(mu,[1 size(z,2)]))))';
    m_temp = repmat(m,[1 size(z,2)]) + K*(z-repmat(mu,[1 size(z,2)]));
    P_temp = (eye(size(P))-K*H)*P;

end