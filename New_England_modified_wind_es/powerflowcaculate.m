%*********************************************************************
%*** New England 10 machine 39 bus 46 line power system *************
%*********************************************************************
function [opdata,opdata_all,YY_load]=powerflowcaculate(busdata,linedata)

% global YY0 PF opdata YY_load; %Y0为不含发电机、负荷的Y阵

%  参数
j=sqrt(-1);


%迭代容许误差
lta=1e-5;

%%%  求解导纳矩阵
nl=linedata(:,1);
nr=linedata(:,2);
R=linedata(:,3);
X=linedata(:,4);
B=linedata(:,5)/2; %%注意，linedata中给定的B为线路总分布电纳
Tap=linedata(:,6);

nbr=length(nl);
nbus=max(max(nl),max(nr));
Z=R+j*X;
y=ones(nbr,1)./Z; %%支路阻抗取倒数

Y=zeros(nbus,nbus);

for k=1:nbr
    if (nl(k)>0)&(nr(k)>0)
        if Tap(k)==0
            Y(nl(k),nr(k))=Y(nl(k),nr(k))-y(k);
        else
            Y(nl(k),nr(k))=Y(nl(k),nr(k))-y(k)/Tap(k);
        end
        Y(nr(k),nl(k))=Y(nl(k),nr(k));
    end
end

for i=1:nbus
    for k=1:nbr
        if nl(k)==i|nr(k)==i
            if (Tap(k)~=0)&(nl(k)==i)
                Y(i,i)=Y(i,i)+y(k)/(Tap(k)*Tap(k))+j*B(k);
            else
                Y(i,i)=Y(i,i)+y(k)+j*B(k);
            end
        end
    end
end

Gs=busdata(:,8);
Bs=busdata(:,9);
Ys=Gs+j*Bs;
for i=1:nbus
    Y(busdata(i,1),busdata(i,1))=Y(busdata(i,1),busdata(i,1))+Ys(i);
end

Y=sparse(Y);

%%%%******************************************
%%% m个PQ节点，n-m-1个PV节点，n点为平衡节点
%%%%******************************************
Busno=busdata(:,10);
PQno=find(busdata(:,10)==3);    
PVno=find(busdata(:,10)==2);
SWno=find(busdata(:,10)==1);
PQVno=[PQno;PVno];
PQVSno=[PQno;PVno;SWno];

m=length(PQno);
n=length(Busno);

%%%按照节点规整节点导纳矩阵
%%%***sy因为10机39节点里面节点顺序就是按照PQ,PV,SW来的，所以这步没影响
YY0=Y;
Y=[YY0(busdata(PQno,1),:);YY0(busdata(PVno,1),:);YY0(busdata(SWno,1),:)];
YY0=Y;
Y=[YY0(:,busdata(PQno,1)) YY0(:,busdata(PVno,1)) YY0(:,busdata(SWno,1))];
YG=real(Y);
YB=imag(Y);


% %初始条件 %%***sy节点功率已注入节点为正
Pg=busdata(PQVno,4);
PL=busdata(PQVno,6);
Ps=Pg-PL;  

Qg=busdata(PQno,5);
QL=busdata(PQno,7);
Qs=Qg-QL;

Vs=busdata(PQVSno,2);
Dels=busdata(PQVSno,3)*pi/180;

P=zeros(n-1,1);
Q=zeros(m,1);

for i=1:n-1
    P(i)=0;
    for k=1:n
        P(i)=P(i)+Vs(i)*Vs(k)*(YG(i,k)*cos(Dels(i)-Dels(k))+YB(i,k)*sin(Dels(i)-Dels(k)));
    end
end

for i=1:m
    Q(i)=0;
    for k=1:n
        Q(i)=Q(i)+Vs(i)*Vs(k)*(YG(i,k)*sin(Dels(i)-Dels(k))-YB(i,k)*cos(Dels(i)-Dels(k)));
    end
end

dPs=Ps-P;
dQs=Qs-Q;
dPQ=[dPs;dQs];

dPQmax=max(abs(dPQ));

flag=0;

%%%牛顿法迭代求解潮流
while (dPQmax>lta)
    
    Jac=zeros(n-1+m);
    JH=zeros(n-1);
    JN=zeros(n-1,m);
    JK=zeros(m,n-1);
    JL=zeros(m,m);
    
    %%% 计算JH
    for i=1:n-1
        for k=1:n-1
            if i==k
                QQs=0;
                for l=1:n
                    QQs=QQs+Vs(i)*Vs(l)*(YG(i,l)*sin(Dels(i)-Dels(l))-YB(i,l)*cos(Dels(i)-Dels(l)));
                end
                JH(i,k)=(Vs(i)^2)*YB(i,k)+QQs;
            else
                JH(i,k)=-Vs(i)*Vs(k)*(YG(i,k)*sin(Dels(i)-Dels(k))-YB(i,k)*cos(Dels(i)-Dels(k)));
            end
        end
    end
    
    %%% 计算JN
    for i=1:n-1
        for k=1:m
            if i==k
                PPs=0;
                for l=1:n
                    PPs=PPs+Vs(i)*Vs(l)*(YG(i,l)*cos(Dels(i)-Dels(l))+YB(i,l)*sin(Dels(i)-Dels(l)));
                end
                JN(i,k)=-(Vs(i)^2)*YG(i,k)-PPs;
            else
                JN(i,k)=-Vs(i)*Vs(k)*(YG(i,k)*cos(Dels(i)-Dels(k))+YB(i,k)*sin(Dels(i)-Dels(k)));
            end
        end
    end
    
    %%% 计算JK
    for i=1:m
        for k=1:n-1
            if i==k
                PPs=0;
                for l=1:n
                    PPs=PPs+Vs(i)*Vs(l)*(YG(i,l)*cos(Dels(i)-Dels(l))+YB(i,l)*sin(Dels(i)-Dels(l)));
                end
                JK(i,k)=(Vs(i)^2)*YG(i,k)-PPs;
            else
                JK(i,k)=Vs(i)*Vs(k)*(YG(i,k)*cos(Dels(i)-Dels(k))+YB(i,k)*sin(Dels(i)-Dels(k)));
            end
        end
    end
    
    %%% 计算JL
    for i=1:m
        for k=1:m
            if i==k
                QQs=0;
                for l=1:n
                    QQs=QQs+Vs(i)*Vs(l)*(YG(i,l)*sin(Dels(i)-Dels(l))-YB(i,l)*cos(Dels(i)-Dels(l)));
                end
                JL(i,k)=(Vs(i)^2)*YB(i,k)-QQs;
            else
                JL(i,k)=-Vs(i)*Vs(k)*(YG(i,k)*sin(Dels(i)-Dels(k))-YB(i,k)*cos(Dels(i)-Dels(k)));
            end
        end
    end
    
    Jac=[JH JN;JK JL];
    dPQ=[dPs;dQs];
    
    dDV=-inv(Jac)*dPQ;
    dD=dDV(1:n-1);
    dVs=diag(Vs(1:m))*dDV(n:n+m-1);
    
    Dels(1:n-1)=Dels(1:n-1)+dD;
    Vs(1:m)=Vs(1:m)+dVs;
    
    %%%%%计算dPQs
    for i=1:n-1
        P(i)=0;
        for k=1:n
            P(i)=P(i)+Vs(i)*Vs(k)*(YG(i,k)*cos(Dels(i)-Dels(k))+YB(i,k)*sin(Dels(i)-Dels(k)));
        end
    end
    
    for i=1:m
        Q(i)=0;
        for k=1:n
            Q(i)=Q(i)+Vs(i)*Vs(k)*(YG(i,k)*sin(Dels(i)-Dels(k))-YB(i,k)*cos(Dels(i)-Dels(k)));
        end
    end
    
    dPs=Ps-P;
    dQs=Qs-Q;
    dPQ=[dPs;dQs];
    
    dPQmax=max(abs(dPQ));
    
    VDelta=[busdata(PQVSno,1) Vs Dels];
    
    flag=flag+1;
    if flag>20
        disp(['***********************************'])
        disp(['There is no solution for power flow'])
        disp(['***********************************'])
        break
    end     
end

%计算平衡节点的注入功率
i=n;
Qsn=0;
for l=1:n
    Qsn=Qsn+Vs(i)*Vs(l)*(YG(i,l)*sin(Dels(i)-Dels(l))-YB(i,l)*cos(Dels(i)-Dels(l)));
end

Psn=0;
for l=1:n
    Psn=Psn+Vs(i)*Vs(l)*(YG(i,l)*cos(Dels(i)-Dels(l))+YB(i,l)*sin(Dels(i)-Dels(l)));
end

%===============平衡节点的视在功率===============
Sn=Psn+j*Qsn;
Vss=Vs.*cos(Dels)+j*Vs.*sin(Dels);
%=======按PQ,PV,Swing排布的节点电压信息===========
%VSS=[编号，电压幅值，相角(°)，向量（复数）]
Vss=Vs.*cos(Dels)+j*Vs.*sin(Dels);
VSS=[busdata(PQVSno,1) abs(Vss)  Dels*180/pi Vss];
% VSS=[busdata(PQVSno,1) abs(Vss)  Dels Vss];

VSS0=[];%% VSS0每列依次为节点编号，电压幅值，相角，复数形式的电压_sy
% for i=1:n
%    for k=1:n
%        if VSS(k,1)==i
%            VSS0=[VSS0;VSS(k,:)];
%        end
%    end
% end
%==============按顺序编号的节点电压信息=============
%VSS0=[编号，电压幅值，相角(°)，向量（复数）]
VSS0=sortrows(VSS,1);

%计算发电机节点的无功功率
QQg=zeros(length(PVno),1);
for i=1:length(PVno);
    QQg(i)=0;
    for l=1:n
        QQg(i)=QQg(i)+Vs(m+i)*Vs(l)*(YG(m+i,l)*sin(Dels(m+i)-Dels(l))-YB(m+i,l)*cos(Dels(m+i)-Dels(l)));
    end
end

QQG=[busdata(PVno,1) QQg]; %QQg算到发电机1~9的无功Q=Qg-QL（sy）

% 计算除平衡节点外所有节点无功Q=Qg-QL
for ii=1:n-1
    Qs(ii)=0;
    for l=1:n
        Qs(ii)=Qs(ii)+Vs(ii)*Vs(l)*(YG(ii,l)*sin(Dels(ii)-Dels(l))-YB(ii,l)*cos(Dels(ii)-Dels(l)));
    end
end

%=================按顺序编号的潮流信息==================
%PF=[编号，电压幅值，相角(°)，向量（复数），Pg,Qg,PL,QL]
PL=busdata(PQVSno,6);
QL=busdata(PQVSno,7);
%%%(sy)PF中不论是平衡节点的P,Q，还是PV节点的Q，最终要求Qg，Pg要加上负荷的。因为Ps=Pg-PL;Qs=Qg-QL
Pg=[busdata(1:38,4);Psn+busdata(39,6)];
Qg=[Qs+busdata(1:38,7);Qsn+busdata(39,7)];
PQ=[Pg Qg PL QL];
disp('      编号               电压幅值                相角(°)              向量（复数）           Pg             Qg             PL           QL')
PF=[VSS0 PQ]

PPL=busdata(:,6);
QQL=busdata(:,7);
VVL=VSS0(:,2);
YYL=(PPL-j*QQL)./(VVL.*VVL);
%%% 将负荷归并到导纳矩阵中去
YY=sparse(YY0+diag(YYL));
YY_load=full(YY);

%=================按顺序编号的发电机信息==================
SG10=busdata(n,6)+j*busdata(n,7)+Sn;
SG=[busdata(PVno,1) busdata(PVno,4) QQg+busdata(PVno,7);n real(SG10) imag(SG10)];
%%%%%%(sy)opdata中不论是平衡节点的P,Q，还是PV节点的Q，最终要求Qg，Pg要加上负荷的。因为Ps=Pg-PL;Qs=Qg-QL
%opdata=[编号，Pg，Qg，电压幅值，相角(°)]
% gen_in_idx=sort([SynMach_in_gidx,EStorage_in_gidx,AggWind_in_gidx]);
% SG=[busdata(gen_in_idx,1) busdata(gen_in_idx,4) QQg+busdata(gen_in_idx,7);n real(SG10) imag(SG10)];
disp("SG")
disp(SG)
disp("VSS")

%%
%=================PQ节点的功率和电压信息==================
PQ_g_idx = PQno;  % PQ节点的索引
PQ_bus_ids = busdata(PQ_g_idx, 1);  % PQ节点编号

% 获取 Ps 和 Qs
PQ_Ps = Ps(1:length(PQ_g_idx));  % 已知 Pg - PL
PQ_Qs = Qs(1:length(PQ_g_idx));  % 已知 Qg - QL

% 取 PL, QL
PQ_PL = busdata(PQ_g_idx, 6);
PQ_QL = busdata(PQ_g_idx, 7);

% 发电功率 Pg = Ps + PL，Qg = Qs + QL
PQ_Pg = PQ_Ps + PQ_PL;
PQ_Qg = PQ_Qs + PQ_QL;

% 电压幅值和相角
PQ_Vmag = VSS0(PQ_bus_ids, 2);  % 幅值
PQ_Vang = VSS0(PQ_bus_ids, 3);  % 相角（°）

% 汇总为矩阵
PQ_opdata = [PQ_bus_ids PQ_Pg PQ_Qg PQ_Vmag PQ_Vang];
% disp("PQ节点信息（含风电等发电源）：编号、Pg、Qg、电压幅值、相角")
% disp(PQ_opdata)

opdata=[SG VSS0([PVno;SWno],2:3)];
opdata_all = [PQ_opdata; opdata];  % opdata 已包含 PV+SW 节点
opdata_all = sortrows(opdata_all, 1);  % 按编号排序
% disp("所有节点发电信息")
% disp(opdata)
%=================按顺序编号的支路信息==================
% Sline1=VSS0(nl,4).*conj((VSS0(nl,4)-VSS0(nr,4)).*y+VSS0(nl,4).*(j*B));
% Sline2=VSS0(nr,4).*conj((VSS0(nr,4)-VSS0(nl,4)).*y+VSS0(nr,4).*(j*B));
% line_Power=[nl nr real(Sline1) imag(Sline1) nr nl real(Sline2) imag(Sline2)];
% Ploss=sum(real(Sline1))+sum(real(Sline2));


end
