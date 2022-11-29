#include<bits/stdc++.h>
using namespace std;
#define MP make_pair
#define F first
#define S second

int n, m;
const int N = 1e5 + 10, H = 30;
int h[N], par[H][N], mx[H][N], root[N];
bool mark[N];
vector<int> t[N];
pair<pair<int, int>, pair<int, int> > e[N];

int get_max(int v, int d){
	int res = 0;
	for(int i = H - 1; i >= 0; i --){
		if((1 << i) <= d){
			d -= (1 << i);
			res = max(res, mx[i][v]);
			v = par[i][v];	
		}
	}
	return res;
}

int get_par(int v, int d){
	for(int i = H - 1; i >= 0; i --){
		if((1 << i) <= d){
			d -= (1 << i);
			v = par[i][v];
		}
	}
	return v;
}

int get_lca(int v, int u){
	if(h[v] > h[u])swap(v, u);
	u = get_par(u, h[u] - h[v]);
	if(u == v)return u;
	for(int i = H - 1; i >= 0; i --){
		if(par[i][v] != par[i][u]){
			v = par[i][v];
			u = par[i][u];
		}
	}
	return par[0][v];
}

void dfs(int v, int p){
	for(int i = 1; i < H; i ++){
		if(h[v] < (1 << i))continue;
		mx[i][v] = max(mx[i - 1][v], mx[i - 1][par[i - 1][v]]);
		par[i][v] = par[i - 1][par[i - 1][v]];
	}
	for(int i = 0; i < (int)t[v].size(); i ++){
		int ind = t[v][i];
		int u = e[ind].S.F + e[ind].S.S - v;
		if(u == p)continue;
		par[0][u] = v;
		mx[0][u] = e[ind].F.F;
		h[u] = h[v] + 1;
		dfs(u, v);
	}
}

int get_root(int v){
	if(root[v] == v)return v;
	else return root[v] = get_root(root[v]);
}

void merge(int v, int u){
	v = get_root(v);
	u = get_root(u);
	root[v] = u;
}

int main(){
	cin >> n >> m;
	for(int i = 0; i < m; i ++){
		int a, b, c;
		cin >> a >> b >> c;
		a--, b--;
		e[i] = MP(MP(c, i), MP(a, b));
	}
	for(int i = 0; i < n; i ++)root[i] = i;
	sort(e, e + m);
	for(int i = 0; i < m; i ++){
		int v = e[i].S.F, u = e[i].S.S;
		if(get_root(v) == get_root(u))continue;
		merge(v, u);
		t[v].push_back(i);
		t[u].push_back(i);
		int index = e[i].F.S;
		mark[index] = true;
	//	cout << index + 1 << endl;
	}
	dfs(0, -1);
	for(int i = 0; i < m; i ++){
		int index = e[i].F.S;
		if(mark[index])continue;
		int v = e[i].S.F, u = e[i].S.S;
		int lc = get_lca(v, u);
		int max_l = get_max(v, h[v] - h[lc]);
		max_l = max(max_l, get_max(u, h[u] - h[lc]));
		if(max_l == e[i].F.F)mark[index] = true;
	}
	for(int i = 0; i < m; i ++){
		cout << 2 - mark[i] << endl;
	}
	return 0;
}
