function [sys,x0,str,ts,simStateCompliance] = TS_net_1_new(t,x,u,flag,busdata,linedata,VSS0,LINE,FBUS,ng,Currentin_idx,amp)
%VSS0初始电压幅值数据（来自潮流结果）；fbus是加入扰动的母线编号，用于某些扰动；ng是发电源的个数；Currentin_idx是；amp是一个导纳扰动参数


persistent YY0 Yff;
persistent YGB YGBd YGBq;
persistent Yng1 Ynn1;
persistent dis
persistent disturb_times amp2
    nBus=size(busdata,1);
    ng2=2*ng;
    ii=1:ng;
    ii2=2*ii;
    ii6=6*ii;
    


%
% The following outlines the general structure of an S-function.
%
switch flag
   
  %%%%%%%%%%%%%%%%%%
  % Initialization %
  %%%%%%%%%%%%%%%%%%
  case 0
    % disturb_times = [ ...
    % 0   + 20*rand, ...
    % 20  + 20*rand, ...
    % 40  + 20*rand, ...
    % 60  + 20*rand, ...
    % 80  + 20*rand];
    % amp2 = sqrt(5) * randn(1,5);
    % fprintf('TS_net_1_new init: disturb_times = %s;disturb = %s', ...
    %     mat2str(disturb_times,6), ...
    %     mat2str(amp2,6));
    [sys,x0,str,ts,simStateCompliance,YY0,Yff,YGB,YGBd,YGBq,Yng1,Ynn1,dis]=mdlInitializeSizes(busdata,linedata,VSS0,LINE,nBus,Currentin_idx,ng,amp);

  %%%%%%%%%%%%%%%
  % Derivatives %
  %%%%%%%%%%%%%%%
  case 1
    sys=[];

  %%%%%%%%%%
  % Update %
  %%%%%%%%%%
  case 2
    sys=[];

  %%%%%%%%%%%
  % Outputs % 输出计算
  %%%%%%%%%%%
  case 3
    sys=mdlOutputs(t,x,u,busdata,linedata,YY0,Yff,FBUS,ng2,ii,ii2,ii6,YGB,YGBd,YGBq,Yng1,Ynn1,Currentin_idx,ng,amp2,disturb_times);

  %%%%%%%%%%%%%%%%%%%%%%%
  % GetTimeOfNextVarHit %   下次采样时间
  %%%%%%%%%%%%%%%%%%%%%%%
  case 4
    sys=mdlGetTimeOfNextVarHit(t,x,u);

  %%%%%%%%%%%%%
  % Terminate %
  %%%%%%%%%%%%%
  case 9
    sys=[];

  %%%%%%%%%%%%%%%%%%%%
  % Unexpected flags %
  %%%%%%%%%%%%%%%%%%%%
  otherwise
    DAStudio.error('Simulink:blocks:unhandledFlag', num2str(flag));

end

% end sfuntmpl

%
%=============================================================================
% mdlInitializeSizes
% Return the sizes, initial conditions, and sample times for the S-function.
%=============================================================================
%
function [sys,x0,str,ts,simStateCompliance,YY0,Yff,YGB,YGBd,YGBq,Yng1,Ynn1,dis]=mdlInitializeSizes(busdata,linedata,VSS0,LINE,nBus,Currentin_idx,ng,amp)

%
% call simsizes for a sizes structure, fill it in and convert it to a
% sizes array.
%
% Note that in this example, the values are hard coded.  This is not a
% recommended practice as the characteristics of the block are typically
% defined by the S-function parameters.
%
sizes = simsizes;

sizes.NumContStates  = 0;
sizes.NumDiscStates  = 0;
sizes.NumOutputs     = 2*ng+1; % 电压包含实轴和虚轴+联络线功率
sizes.NumInputs      = 6*ng+1; % 输入包含clock+电流*2+导纳（4维）
sizes.DirFeedthrough = 1;
sizes.NumSampleTimes = 1;   % at least one sample time is needed

sys = simsizes(sizes);

%
% initialize the initial conditions
%
x0  = [];

%
% str is always an empty matrix
%
str = [];

%
% initialize the array of sample times
%
ts  = [-1 0];

% Specify the block simStateCompliance. The allowed values are:
%    'UnknownSimState', < The default setting; warn and assume DefaultSimState
%    'DefaultSimState', < Same sim state as a built-in block
%    'HasNoSimState',   < No sim state
%    'DisallowSimState' < Error out when saving or restoring the model sim state
simStateCompliance = 'UnknownSimState';


    bus     = busdata;
    branch  = linedata;
    baseMVA = 100;
dis = zeros(1,16);
dis(4) = 3*randn + j*3*randn;
dis(9) = 3*randn;
dis(13) = 3*randn;
dis(14) = 3*randn;
dis(15) = 3*randn;
%% constants
nb = size(bus, 1);          %% number of buses
nl = size(branch, 1);       %% number of lines

%% for each branch, compute the elements of the branch admittance matrix where
%%
%%      | If |   | Yff  Yft |   | Vf |
%%      |    | = |          | * |    |
%%      | It |   | Ytf  Ytt |   | Vt |
%% 有效线路选取
stat = branch{:, 'Brc_status'};                    %% ones at in-service branches
Ys = stat ./ (branch{:, 'R'} + 1j * branch{:, 'X'});  %% series admittance
Bc = stat .* branch{:, 'B'};                           %% line charging susceptance
%% 分接变压器处理
tap = ones(nl, 1);                              %% default tap ratio = 1
i = find(branch{:, 'Tap'});                       %% indices of non-zero tap ratios
tap(i) = branch{i, 'Tap'};                        %% assign non-zero tap ratios
% tap = tap .* exp(1j*pi/180 * branch(:, SHIFT)); %% add phase shifters
%% 计算四个支路导纳分量：
Ytt = Ys + 1j*Bc/2;
Yff = Ytt ./ (tap .* conj(tap));
Yft = - Ys ./ conj(tap);
Ytf = - Ys ./ tap;

%% compute shunt admittance
%% if Psh is the real power consumed by the shunt at V = 1.0 p.u.
%% and Qsh is the reactive power injected by the shunt at V = 1.0 p.u.
%% then Psh - j Qsh = V * conj(Ysh * V) = conj(Ysh) = Gs - j Bs,
%% i.e. Ysh = Psh + j Qsh, so ...
Ysh = (bus{:, 'GS'} + 1j * bus{:, 'BS'}) / baseMVA; %% vector of shunt admittances

%% bus indices
f = branch{:, 'F_Bus'};                           %% list of "from" buses
t = branch{:, 'T_Bus'};                           %% list of "to" buses

%% for best performance, choose method based on MATLAB vs Octave and size
if nb < 300 || have_fcn('octave')   %% small case OR running on Octave
    %% build Yf and Yt such that Yf * V is the vector of complex branch currents injected
    %% at each branch's "from" bus, and Yt is the same for the "to" bus end
    i = [1:nl 1:nl]';                           %% double set of row indices
    Yf = sparse(i, [f; t], [Yff; Yft], nl, nb);
    Yt = sparse(i, [f; t], [Ytf; Ytt], nl, nb);

    %% build Ybus
    Ybus = sparse([f;f;t;t], [f;t;f;t], [Yff;Yft;Ytf;Ytt], nb, nb) + ... %% branch admittances
            sparse(1:nb, 1:nb, Ysh, nb, nb);        %% shunt admittance
else                                %% large case running on MATLAB
    %% build connection matrices
    Cf = sparse(1:nl, f, ones(nl, 1), nl, nb);      %% connection matrix for line & from buses
    Ct = sparse(1:nl, t, ones(nl, 1), nl, nb);      %% connection matrix for line & to buses

    %% build Yf and Yt such that Yf * V is the vector of complex branch currents injected
    %% at each branch's "from" bus, and Yt is the same for the "to" bus end
    Yf = sparse(1:nl, 1:nl, Yff, nl, nl) * Cf + sparse(1:nl, 1:nl, Yft, nl, nl) * Ct;
    Yt = sparse(1:nl, 1:nl, Ytf, nl, nl) * Cf + sparse(1:nl, 1:nl, Ytt, nl, nl) * Ct;

    %% build Ybus
    Ybus = Cf' * Yf + Ct' * Yt + ...            %% branch admittances
            sparse(1:nb, 1:nb, Ysh, nb, nb);    %% shunt admittance
end

    nl=linedata{:,'F_Bus'};%
    nr=linedata{:,'T_Bus'};%

    nbr=length(nl);%
    nbus=max(max(nl),max(nr));%
     
    Ybus=sparse(Ybus);%
    
    YY=Ybus;
    %% 计算负荷等效导纳 Y=S/V^2
    PPL=busdata{:,'PD'};%PPL=sin(wt)
    QQL=busdata{:,'QD'};
    VVL=VSS0{:,'Vm'};
    YYL=(PPL-1j*QQL)./(VVL.*VVL);
    
    YY=sparse(YY+diag(YYL));%
    % 使得Ybus包含负载效应
    YY0=YY;%
    
    
%%     
%

    
    R=LINE{1,'R'};
    X=LINE{1,'X'};
    B=LINE{1,'B'}/2;

    F=LINE{1,'F_Bus'};
    T=LINE{1,'T_Bus'};
    
    Yff=zeros(nbus,nbus);
    Yff(F,F)=1/(R+1j*X)+1j*B;
    Yff(F,T)=-1/(R+1j*X);
    Yff(T,F)=-1/(R+1j*X);
    Yff(T,T)=1/(R+1j*X)+1j*B;
    Yff=sparse(Yff);
    

    ng2=2*ng;
    ii=1:ng;
    ii2=2*ii;
%% 
    Y=YY0;
    
    
    
%%         
    Nidx=1:nBus;
    Nidx(Currentin_idx)=[];%Nidx表示没有发电源的母线
    Ygg=Y(Currentin_idx,Currentin_idx);Ygn=Y(Currentin_idx,Nidx);%Ygg发电源与发电源之间的导纳连接，Yng非发电源与发电源之间的导纳连接
    Yng=Y(Nidx,Currentin_idx);Ynn=Y(Nidx,Nidx);
    Yng1=Yng;Ynn1=Ynn;
    Yred=Ygg-Ygn/Ynn*Yng;
    
    YGB=zeros(ng2,ng2);
    YGB(ii2-1,ii2-1)=real(Yred(ii,ii));
    YGB(ii2-1,ii2)  =-imag(Yred(ii,ii));
    YGB(ii2,ii2-1)  =imag(Yred(ii,ii));
    YGB(ii2,ii2)    =real(Yred(ii,ii));%YGB是基准状态（无扰动）
%% 
    YGBd=YGB;
    YGBq=YGB;
% if fbus~=999
%     Y(fbus,fbus)=Y(fbus,fbus)+amp; % 相当于加入并联定导纳负荷
% 
% 
%     Nidx=1:nBus; % 
%     Nidx(Currentin_idx)=[]; 
%     Ygg=Y(Currentin_idx,Currentin_idx);Ygn=Y(Currentin_idx,Nidx); 
%     Yng=Y(Nidx,Currentin_idx);Ynn=Y(Nidx,Nidx);
%     Yng1=Yng;Ynn1=Ynn;
%     %%      | Ig |   | Ygg  Ygn |   | Vg |
%     %%      |    | = |          | * |    |
%     %%      | In |   | Yng  Ynn |   | Vn |
%     Yred=Ygg-Ygn/Ynn*Yng;
% 
%     YGBd=zeros(ng2,ng2);
%     YGBd(ii2-1,ii2-1)=real(Yred(ii,ii));
%     YGBd(ii2-1,ii2)  =-imag(Yred(ii,ii));
%     YGBd(ii2,ii2-1)  =imag(Yred(ii,ii));
%     YGBd(ii2,ii2)    =real(Yred(ii,ii));%YGBd母线 fbus 被扰动（注入电导扰动 amp）
% %%
%     Y=Y-Yff;
%     Nidx=1:nBus;
%     Nidx(Currentin_idx)=[];%
%     Ygg=Y(Currentin_idx,Currentin_idx);Ygn=Y(Currentin_idx,Nidx);
%     Yng=Y(Nidx,Currentin_idx);Ynn=Y(Nidx,Nidx);
%     Yred=Ygg-Ygn/Ynn*Yng;
% 
%     YGBq=zeros(ng2,ng2);
%     YGBq(ii2-1,ii2-1)=real(Yred(ii,ii));
%     YGBq(ii2-1,ii2)  =-imag(Yred(ii,ii));
%     YGBq(ii2,ii2-1)  =imag(Yred(ii,ii));
%     YGBq(ii2,ii2)    =real(Yred(ii,ii));%YGBq同时断开线路 LINE 和注入电导扰动
%     % 这是将复数导纳矩阵转化为实数矩阵的经典方法：
%     % 将每个复数表示为 2x2 实数块，用于后续实数解算器仿真。
% end

% end mdlInitializeSizes

%
%=============================================================================
% mdlOutputs
% Return the block outputs.
%=============================================================================
%
function sys=mdlOutputs(t,x,u,busdata,linedata,YY0,Yff,FBUS,ng2,ii,ii2,ii6,YGB,YGBd,YGBq,Yng1,Ynn1,Currentin_idx,ng,amp2,disturb_times)


nBus=size(busdata,1);
Y=YY0;
%%            ====================

t1=0.1;    %5
t2=50000;    

t3=50000;  

t0=1; %施加扰动的时间

amp1=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,5,0];%施加负荷扰动
Y=YY0;
t01=300;
if (u(1)>=t0)
    for i = 1:16
        fbus=i;
        Y(fbus,fbus)=Y(fbus,fbus)+amp1(i); % 相当于加入并联定导纳负荷
    end
    Nidx=1:nBus; % 
    Nidx(Currentin_idx)=[]; 
    Ygg1=Y(Currentin_idx,Currentin_idx);Ygn1=Y(Currentin_idx,Nidx); 
    Yng1=Y(Nidx,Currentin_idx);Ynn1=Y(Nidx,Nidx);
    
    Yred=Ygg1-Ygn1/Ynn1*Yng1;
    YGBd=zeros(ng2,ng2);
    YGBd(ii2-1,ii2-1)=real(Yred(ii,ii));
    YGBd(ii2-1,ii2)  =-imag(Yred(ii,ii));
    YGBd(ii2,ii2-1)  =imag(Yred(ii,ii));
    YGBd(ii2,ii2)    =real(Yred(ii,ii));%YGBd母线 fbus 被扰动（注入电导扰动 amp）
end
% for k = 1:length(disturb_times)
%     if (u(1)>=disturb_times(k))
% 
%         Y(15,15)=Y(15,15)+amp2(k); % 相当于加入并联定导纳负荷
%         Nidx=1:nBus; % 
%         Nidx(Currentin_idx)=[]; 
%         Ygg1=Y(Currentin_idx,Currentin_idx);Ygn1=Y(Currentin_idx,Nidx); 
%         Yng1=Y(Nidx,Currentin_idx);Ynn1=Y(Nidx,Nidx);
% 
%         Yred=Ygg1-Ygn1/Ynn1*Yng1;
%         YGBd=zeros(ng2,ng2);
%         YGBd(ii2-1,ii2-1)=real(Yred(ii,ii));
%         YGBd(ii2-1,ii2)  =-imag(Yred(ii,ii));
%         YGBd(ii2,ii2-1)  =imag(Yred(ii,ii));
%         YGBd(ii2,ii2)    =real(Yred(ii,ii));%YGBd母线 fbus 被扰动（注入电导扰动 amp）
%     end
% end
if (u(1)>=t01)
    Y(fbus,fbus)=Y(fbus,fbus)-amp1; % 相当于加入并联定导纳负荷

    Nidx=1:nBus; % 
    Nidx(Currentin_idx)=[]; 
    Ygg1=Y(Currentin_idx,Currentin_idx);Ygn1=Y(Currentin_idx,Nidx); 
    Yng1=Y(Nidx,Currentin_idx);Ynn1=Y(Nidx,Nidx);

    Yred=Ygg1-Ygn1/Ynn1*Yng1;
    YGBd=zeros(ng2,ng2);
    YGBd(ii2-1,ii2-1)=real(Yred(ii,ii));
    YGBd(ii2-1,ii2)  =-imag(Yred(ii,ii));
    YGBd(ii2,ii2-1)  =imag(Yred(ii,ii));
    YGBd(ii2,ii2)    =real(Yred(ii,ii));%YGBd母线 fbus 被扰动（注入电导扰动 amp）
end

amp2=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0];%施加负荷扰动
t02=25;
if (u(1)>=t02)

    for i = 1:16
        fbus=i;
        Y(fbus,fbus)=Y(fbus,fbus)+amp2(i); % 相当于加入并联定导纳负荷
    end
    Nidx=1:nBus; % 
    Nidx(Currentin_idx)=[]; 
    Ygg1=Y(Currentin_idx,Currentin_idx);Ygn1=Y(Currentin_idx,Nidx); 
    Yng1=Y(Nidx,Currentin_idx);Ynn1=Y(Nidx,Nidx);

    Yred=Ygg1-Ygn1/Ynn1*Yng1;
    YGBd=zeros(ng2,ng2);
    YGBd(ii2-1,ii2-1)=real(Yred(ii,ii));
    YGBd(ii2-1,ii2)  =-imag(Yred(ii,ii));
    YGBd(ii2,ii2-1)  =imag(Yred(ii,ii));
    YGBd(ii2,ii2)    =real(Yred(ii,ii));%YGBd母线 fbus 被扰动（注入电导扰动 amp）
end
% 基于当前扰动时间，选择YGBd还是YGBq
if (u(1)>=t1)&&(u(1)<t2)
    
    YGB=YGBd;
    
end

if (u(1)>=t2)&&(u(1)<t3) 
    
    YGB=YGBq;
         
end



%%%%

YGB(  (ng2+1).*(ii2-1)-ng2  )      =   YGB(  (ng2+1).*(ii2-1)-ng2  ) +u(ii6-2).'; % 线路导纳加上发电源导纳
YGB( (ng2+1).*(ii2)-ng2-1   )      =   YGB( (ng2+1).*(ii2)-ng2-1   ) +u(ii6).';
YGB( (ng2+1).*(ii2-1)-ng2+1 )      =   YGB( (ng2+1).*(ii2-1)-ng2+1 ) +u(ii6+1).';
YGB( (ng2+1).*(ii2)-ng2     )      =   YGB( (ng2+1).*(ii2)-ng2     ) +u(ii6-1).';


%%%%
 Ixy=zeros(ng2,1);
 Ixy([ii2-1,ii2])=u([ii6-4,ii6-3]);


Vxy=YGB\Ixy; % 左除运算符，Vxy=YGB-1*Ixy → YGB*Vxy=Ixy


%%%%
% Vg=zeros(ng,1);
% for i=1:ng
%     Vg(i)=Vxy(2*i-1)+1j*Vxy(2*i);
% end

%Vn=-Ynn\Yng*Vg;%
%Vnang=angle(Vn);
%计算联络线功率
% 1. 构造电源节点电压（由仿真输入电流计算）
Vg = Vxy(1:2:end) + 1j * Vxy(2:2:end);  % ng x 1
% disp("size(Yng):"), disp(size(Ygg))
% disp("size(Yng):"), disp(size(Yng))
% disp("size(Vg):"), disp(size(Vg))

% 2. 还原网络中其他节点电压（通过 Ynn 解耦推回）
% 使用初始化阶段保存的 Ynn 和 Yng
Vn = -Ynn1 \ (Yng1 * Vg);  % 非发电源节点的电压（n-ng 个）

% 3. 组装全电压向量 V_bus
V_bus = zeros(nBus,1);
V_bus(Currentin_idx) = Vg;
V_bus(setdiff(1:nBus,Currentin_idx)) = Vn;
% 读取第 k 条支路数据
k = 7; % 第几条线路
f_idx = linedata{k, 'F_Bus'};  % 起点母线编号
t_idx = linedata{k, 'T_Bus'};  % 终点母线编号
R = linedata{k, 'R'};
X = linedata{k, 'X'};
B = linedata{k, 'B'};  % total shunt susceptance

Z = R + 1j * X;
Y = 1/Z;
B_shunt = 1j * B / 2;

% 线路两端电压
Vf = V_bus(f_idx);
Vt = V_bus(t_idx);

% 起点电流（含 shunt）
If = Y * (Vf - Vt) + B_shunt * Vf;

% 起点注入功率
S_from = Vf * conj(If);
P_from = real(S_from);
Q_from = imag(S_from);

deltaP = P_from - 1.77;  % 当前功率偏差


sys = [Vxy;deltaP];

% end mdlOutputs

%
%=============================================================================
% mdlGetTimeOfNextVarHit
% Return the time of the next hit for this block.  Note that the result is
% absolute time.  Note that this function is only used when you specify a
% variable discrete-time sample time [-2 0] in the sample time array in
% mdlInitializeSizes.
%=============================================================================
%
function sys=mdlGetTimeOfNextVarHit(t,x,u)

sampleTime = 1;    %  Example, set the next hit to be one second later.
sys = t + sampleTime;

% end mdlGetTimeOfNextVarHit

