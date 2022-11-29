#include<bits/stdc++.h>
using namespace std;

const int N = 1000;

double x[N], y[N];
int w[N][N], par[N];

bool check_dis(int a, int b){
	double dx = x[a] - x[b], dy = y[a] - y[b];
	if(dx * dx + dy * dy <= 10000)return true;
	return false;
}

void dfs(int v, int n){
	for(int i = 0; i < 2 * n; i ++){
		if(w[v][i] == 0 || par[i] != -1)continue;
		par[i] = v;
		dfs(i, n);
	}
}

int max_flow(int n){
	int res = 0;
	while(true){
		fill(par, par + N, -1);
		dfs(0, n);
		if(par[n - 1] == -1)break;
		res ++;
		int cur = n - 1;
		while(cur != 0){
			w[par[cur]][cur] = 0;
			w[cur][par[cur]] = 1;
			cur = par[cur];
		}
	}
	return res;
}

int main(){
	int n;
	cin >> n;
	cin >> x[n + 1] >> y[n + 1];
	for(int i = 1; i <= n; i ++){
		cin >> x[i] >> y[i];
	}
	int m = n + 2;
	for(int i = 1; i < m - 1; i ++){
		w[i][i + m] = 1;
	}
	for(int i = 1; i < m - 1; i ++){
		for(int j = 1; j < m; j ++){
			if(x[i] >= x[j])continue;
			if(check_dis(i, j))w[i + m][j] = 1;
		}
	}
	for(int i = 1; i < m; i ++){
		if(x[0] >= x[i])continue;
		if(check_dis(0, i))w[0][i] = 1;
	}
	cout << max_flow(m);	
	return 0;
}
