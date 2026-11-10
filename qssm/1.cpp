#include <iostream>
#include <vector>

using namespace std;

int r;
int n, m;
vector<vector<int>> g;
bool v[105];

void f(int x, int y) {
    if (x == n - 1 && y == m - 1) {
        r++;
        return;
    }
    if (x + 1 < n) {
        int c = g[x + 1][y];
        if (!v[c]) {
            v[c] = true;
            f(x + 1, y);
            v[c] = false;
        }
    }
    if (y + 1 < m) {
        int c = g[x][y + 1];
        if (!v[c]) {
            v[c] = true;
            f(x, y + 1);
            v[c] = false;
        }
    }
}

int countColorfulPaths(int N, int M, vector<vector<int>> grid) {
    n = N;
    m = M;
    g = grid;
    r = 0;
    for (int i = 0; i < 105; i++) v[i] = false;
    if (n == 0 || m == 0) return 0;
    v[g[0][0]] = true;
    f(0, 0);
    return r;
}

int main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    int N;
    cin >> N;
    int M;
    cin >> M;
    vector<vector<int>> grid(N, vector<int>(M));
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < M; j++) {
            cin >> grid[i][j];
        }
    }
    int o = countColorfulPaths(N, M, grid);
    cout << o;
    return 0;
}