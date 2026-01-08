import React, { useState } from 'react';
import './App.css';

// Update to use the Render backend URL (with env override for flexibility)
const API_BASE = process.env.REACT_APP_BACKEND_URL || process.env.REACT_APP_API_BASE || 'https://algorithm-analyze-dyas.onrender.com';

const ALGORITHMS = [
  { label: 'Merge Sort (with Tree)', value: 'merge-sort' },
  { label: 'Quick Sort', value: 'quick-sort' },
  { label: 'Selection Sort', value: 'selection-sort' },
  { label: 'Topological Sort (Vertex Removal)', value: 'topo-sort' },
  { label: 'Floyd-Warshall (Shortest Paths)', value: 'floyd-warshall' },
  { label: "Warshall's Algorithm (Transitive Closure)", value: 'warshall' },
  { label: 'Greedy (Activity Selection)', value: 'activity-selection' },
  { label: 'Knapsack (DP)', value: 'knapsack' },
  { label: "Prim's Algorithm (MST)", value: 'prims' },
  { label: "Kruskal's Algorithm (MST)", value: 'kruskal' },
  { label: "Dijkstra's Algorithm (SSSP)", value: 'dijkstra' },
];

const ALGO_INFO = {
  'merge-sort': {
    name: 'Merge Sort',
    time: 'O(n log n)',
    desc: 'Merge Sort is a divide-and-conquer algorithm that recursively splits the array into halves, sorts each half, and merges them back together.'
  },
  'quick-sort': {
    name: 'Quick Sort',
    time: 'O(n log n) average, O(n^2) worst',
    desc: 'Quick Sort is a divide-and-conquer algorithm that partitions the array around a pivot, recursively sorting the partitions.'
  },
  'selection-sort': {
    name: 'Selection Sort',
    time: 'O(n^2)',
    desc: 'Selection Sort repeatedly selects the minimum element from the unsorted part and moves it to the sorted part.'
  },
  'topo-sort': {
    name: 'Topological Sort (Kahn’s Algorithm)',
    time: 'O(V + E)',
    desc: 'Topological Sort orders the vertices of a directed acyclic graph (DAG) such that for every directed edge u → v, u comes before v.'
  },
  'floyd-warshall': {
    name: 'Floyd-Warshall',
    time: 'O(n^3)',
    desc: 'Floyd-Warshall finds shortest paths between all pairs of vertices in a weighted graph (with positive or negative edge weights, but no negative cycles).'
  },
  'warshall': {
    name: 'Warshall’s Algorithm',
    time: 'O(n^3)',
    desc: 'Warshall’s Algorithm computes the transitive closure of a directed graph, determining reachability between all pairs of vertices.'
  },
  'activity-selection': {
    name: 'Activity Selection (Greedy)',
    time: 'O(n log n)',
    desc: 'The Activity Selection problem selects the maximum number of non-overlapping activities, solved efficiently using a greedy approach.'
  },
  'knapsack': {
    name: 'Knapsack (0/1, DP)',
    time: 'O(nW)',
    desc: 'Solves the 0/1 Knapsack problem using dynamic programming. Finds the max profit for given weights, profits, and capacity.'
  },
  'prims': {
    name: "Prim's Algorithm (MST)",
    time: 'O(E log V)',
    desc: 'Finds a minimum spanning tree (MST) of a weighted undirected graph using a greedy approach.'
  },
  'kruskal': {
    name: "Kruskal's Algorithm (MST)",
    time: 'O(E log V)',
    desc: 'Finds a minimum spanning tree (MST) by sorting edges and adding them without creating cycles.'
  },
  'dijkstra': {
    name: "Dijkstra's Algorithm (SSSP)",
    time: 'O(E log V)',
    desc: 'Finds the shortest path from a source node to all other nodes in a weighted graph (no negative weights).'
  },
};

// Add code snippets for each algorithm in C, C++, Python
const ALGO_CODE = {
  'merge-sort': {
    python: `def merge_sort(arr):
    if len(arr) > 1:
        mid = len(arr) // 2
        L = arr[:mid]
        R = arr[mid:]
        merge_sort(L)
        merge_sort(R)
        i = j = k = 0
        while i < len(L) and j < len(R):
            if L[i] < R[j]:
                arr[k] = L[i]
                i += 1
            else:
                arr[k] = R[j]
                j += 1
            k += 1
        while i < len(L):
            arr[k] = L[i]
            i += 1
            k += 1
        while j < len(R):
            arr[k] = R[j]
            j += 1
            k += 1`,
    cpp: `void merge(int arr[], int l, int m, int r) {
    int n1 = m - l + 1;
    int n2 = r - m;
    int L[n1], R[n2];
    for (int i = 0; i < n1; i++) L[i] = arr[l + i];
    for (int j = 0; j < n2; j++) R[j] = arr[m + 1 + j];
    int i = 0, j = 0, k = l;
    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) arr[k++] = L[i++];
        else arr[k++] = R[j++];
    }
    while (i < n1) arr[k++] = L[i++];
    while (j < n2) arr[k++] = R[j++];
}
void mergeSort(int arr[], int l, int r) {
    if (l < r) {
        int m = l + (r - l) / 2;
        mergeSort(arr, l, m);
        mergeSort(arr, m + 1, r);
        merge(arr, l, m, r);
    }
}`,
    c: `void merge(int arr[], int l, int m, int r) {
    int n1 = m - l + 1;
    int n2 = r - m;
    int L[n1], R[n2];
    for (int i = 0; i < n1; i++) L[i] = arr[l + i];
    for (int j = 0; j < n2; j++) R[j] = arr[m + 1 + j];
    int i = 0, j = 0, k = l;
    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) arr[k++] = L[i++];
        else arr[k++] = R[j++];
    }
    while (i < n1) arr[k++] = L[i++];
    while (j < n2) arr[k++] = R[j++];
}
void mergeSort(int arr[], int l, int r) {
    if (l < r) {
        int m = l + (r - l) / 2;
        mergeSort(arr, l, m);
        mergeSort(arr, m + 1, r);
        merge(arr, l, m, r);
    }
}`
  },
  'quick-sort': {
    python: `def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    else:
        pivot = arr[-1]
        left = [x for x in arr[:-1] if x <= pivot]
        right = [x for x in arr[:-1] if x > pivot]
        return quick_sort(left) + [pivot] + quick_sort(right)`,
    cpp: `int partition(int arr[], int low, int high) {
    int pivot = arr[high];
    int i = (low - 1);
    for (int j = low; j <= high - 1; j++) {
        if (arr[j] <= pivot) {
            i++;
            swap(&arr[i], &arr[j]);
        }
    }
    swap(&arr[i + 1], &arr[high]);
    return (i + 1);
}
void quickSort(int arr[], int low, int high) {
    if (low < high) {
        int pi = partition(arr, low, high);
        quickSort(arr, low, pi - 1);
        quickSort(arr, pi + 1, high);
    }
}`,
    c: `int partition(int arr[], int low, int high) {
    int pivot = arr[high];
    int i = (low - 1);
    for (int j = low; j <= high - 1; j++) {
        if (arr[j] <= pivot) {
            i++;
            int temp = arr[i]; arr[i] = arr[j]; arr[j] = temp;
        }
    }
    int temp = arr[i + 1]; arr[i + 1] = arr[high]; arr[high] = temp;
    return (i + 1);
}
void quickSort(int arr[], int low, int high) {
    if (low < high) {
        int pi = partition(arr, low, high);
        quickSort(arr, low, pi - 1);
        quickSort(arr, pi + 1, high);
    }
}`
  },
  'selection-sort': {
    python: `def selection_sort(arr):
    n = len(arr)
    for i in range(n):
        min_idx = i
        for j in range(i+1, n):
            if arr[j] < arr[min_idx]:
                min_idx = j
        arr[i], arr[min_idx] = arr[min_idx], arr[i]`,
    cpp: `void selectionSort(int arr[], int n) {
    int i, j, min_idx;
    for (i = 0; i < n-1; i++) {
        min_idx = i;
        for (j = i+1; j < n; j++)
            if (arr[j] < arr[min_idx])
                min_idx = j;
        swap(&arr[min_idx], &arr[i]);
    }
}`,
    c: `void selectionSort(int arr[], int n) {
    int i, j, min_idx;
    for (i = 0; i < n-1; i++) {
        min_idx = i;
        for (j = i+1; j < n; j++)
            if (arr[j] < arr[min_idx])
                min_idx = j;
        int temp = arr[min_idx]; arr[min_idx] = arr[i]; arr[i] = temp;
    }
}`
  },
  'topo-sort': {
    python: `def topo_sort(graph):
    visited = set()
    stack = []
    def dfs(v):
        visited.add(v)
        for neighbor in graph[v]:
            if neighbor not in visited:
                dfs(neighbor)
        stack.append(v)
    for v in graph:
        if v not in visited:
            dfs(v)
    stack.reverse()
    return stack`,
    cpp: `void topoSortUtil(int v, vector<bool>& visited, stack<int>& Stack, vector<vector<int>>& graph) {
    visited[v] = true;
    for (int i : graph[v])
        if (!visited[i])
            topoSortUtil(i, visited, Stack, graph);
    Stack.push(v);
}
vector<int> topoSort(int V, vector<vector<int>>& graph) {
    stack<int> Stack;
    vector<bool> visited(V, false);
    for (int i = 0; i < V; i++)
        if (!visited[i])
            topoSortUtil(i, visited, Stack, graph);
    vector<int> order;
    while (!Stack.empty()) {
        order.push_back(Stack.top());
        Stack.pop();
    }
    return order;
}`,
    c: `void topoSortUtil(int v, int visited[], int* stack, int* stackTop, int** graph, int V) {
    visited[v] = 1;
    for (int i = 0; i < V; i++)
        if (graph[v][i] && !visited[i])
            topoSortUtil(i, visited, stack, stackTop, graph, V);
    stack[(*stackTop)++] = v;
}
void topoSort(int V, int** graph) {
    int* stack = malloc(V * sizeof(int));
    int stackTop = 0;
    int* visited = calloc(V, sizeof(int));
    for (int i = 0; i < V; i++)
        if (!visited[i])
            topoSortUtil(i, visited, stack, &stackTop, graph, V);
    for (int i = stackTop - 1; i >= 0; i--)
        printf("%d ", stack[i]);
    free(stack); free(visited);
}`
  },
  'floyd-warshall': {
    python: `def floyd_warshall(graph):
    n = len(graph)
    dist = [row[:] for row in graph]
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i][j] > dist[i][k] + dist[k][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
    return dist`,
    cpp: `void floydWarshall(int graph[][V]) {
    int dist[V][V], i, j, k;
    for (i = 0; i < V; i++)
        for (j = 0; j < V; j++)
            dist[i][j] = graph[i][j];
    for (k = 0; k < V; k++)
        for (i = 0; i < V; i++)
            for (j = 0; j < V; j++)
                if (dist[i][j] > dist[i][k] + dist[k][j])
                    dist[i][j] = dist[i][k] + dist[k][j];
}`,
    c: `void floydWarshall(int graph[][V]) {
    int dist[V][V], i, j, k;
    for (i = 0; i < V; i++)
        for (j = 0; j < V; j++)
            dist[i][j] = graph[i][j];
    for (k = 0; k < V; k++)
        for (i = 0; i < V; i++)
            for (j = 0; j < V; j++)
                if (dist[i][j] > dist[i][k] + dist[k][j])
                    dist[i][j] = dist[i][k] + dist[k][j];
}`
  },
  'warshall': {
    python: `def warshall(graph):
    n = len(graph)
    closure = [row[:] for row in graph]
    for k in range(n):
        for i in range(n):
            for j in range(n):
                closure[i][j] = closure[i][j] or (closure[i][k] and closure[k][j])
    return closure`,
    cpp: `void warshall(int graph[][V]) {
    int closure[V][V], i, j, k;
    for (i = 0; i < V; i++)
        for (j = 0; j < V; j++)
            closure[i][j] = graph[i][j];
    for (k = 0; k < V; k++)
        for (i = 0; i < V; i++)
            for (j = 0; j < V; j++)
                closure[i][j] = closure[i][j] || (closure[i][k] && closure[k][j]);
}`,
    c: `void warshall(int graph[][V]) {
    int closure[V][V], i, j, k;
    for (i = 0; i < V; i++)
        for (j = 0; j < V; j++)
            closure[i][j] = graph[i][j];
    for (k = 0; k < V; k++)
        for (i = 0; i < V; i++)
            for (j = 0; j < V; j++)
                closure[i][j] = closure[i][j] || (closure[i][k] && closure[k][j]);
}`
  },
  'activity-selection': {
    python: `def activity_selection(activities):
    activities.sort(key=lambda x: x[1])
    selected = []
    last_end = -1
    for start, end in activities:
        if start >= last_end:
            selected.append((start, end))
            last_end = end
    return selected`,
    cpp: `vector<pair<int, int>> activitySelection(vector<pair<int, int>>& acts) {
    sort(acts.begin(), acts.end(), [](auto& a, auto& b) { return a.second < b.second; });
    vector<pair<int, int>> selected;
    int last_end = -1;
    for (auto& act : acts) {
        if (act.first >= last_end) {
            selected.push_back(act);
            last_end = act.second;
        }
    }
    return selected;
}`,
    c: `void activitySelection(int start[], int end[], int n) {
    int i, j;
    printf("Selected Activities:\n");
    i = 0;
    printf("(%d, %d) ", start[i], end[i]);
    for (j = 1; j < n; j++) {
        if (start[j] >= end[i]) {
            printf("(%d, %d) ", start[j], end[j]);
            i = j;
        }
    }
}`
  },
  'knapsack': {
    python: `def knapsack(weights, profits, capacity):
    n = len(weights)
    dp = [[0 for _ in range(capacity + 1)] for _ in range(n + 1)]
    for i in range(1, n + 1):
        for w in range(1, capacity + 1):
            if weights[i - 1] <= w:
                dp[i][w] = max(dp[i - 1][w], dp[i - 1][w - weights[i - 1]] + profits[i - 1])
            else:
                dp[i][w] = dp[i - 1][w]
    return dp[n][capacity], dp`,
    cpp: `int knapsack(int weights[], int profits[], int n, int capacity) {
    int dp[n + 1][capacity + 1];
    for (int i = 0; i <= n; i++) {
        for (int w = 0; w <= capacity; w++) {
            if (i == 0 || w == 0)
                dp[i][w] = 0;
            else if (weights[i - 1] <= w)
                dp[i][w] = max(dp[i - 1][w], dp[i - 1][w - weights[i - 1]] + profits[i - 1]);
            else
                dp[i][w] = dp[i - 1][w];
        }
    }
    return dp[n][capacity];
}`,
    c: `int knapsack(int weights[], int profits[], int n, int capacity) {
    int dp[n + 1][capacity + 1];
    for (int i = 0; i <= n; i++) {
        for (int w = 0; w <= capacity; w++) {
            if (i == 0 || w == 0)
                dp[i][w] = 0;
            else if (weights[i - 1] <= w)
                dp[i][w] = max(dp[i - 1][w], dp[i - 1][w - weights[i - 1]] + profits[i - 1]);
            else
                dp[i][w] = dp[i - 1][w];
        }
    }
    return dp[n][capacity];
}`
  },
  'prims': {
    python: `def prims(graph):
    n = len(graph)
    key = [float('inf')] * n
    parent = [-1] * n
    mstSet = [False] * n
    key[0] = 0
    for _ in range(n):
        u = -1
        for v in range(n):
            if not mstSet[v] and (u == -1 or key[v] < key[u]):
                u = v
        mstSet[u] = True
        for v in range(n):
            if graph[u][v] > 0 and not mstSet[v] and graph[u][v] < key[v]:
                key[v] = graph[u][v]
                parent[v] = u
    return parent, key`,
    cpp: `void prims(int graph[][V]) {
    int key[V], parent[V];
    bool mstSet[V];
    for (int i = 0; i < V; i++)
        key[i] = INT_MAX, mstSet[i] = false;
    key[0] = 0;
    parent[0] = -1;
    for (int count = 0; count < V - 1; count++) {
        int u = -1;
        for (int v = 0; v < V; v++)
            if (!mstSet[v] && (u == -1 || key[v] < key[u]))
                u = v;
        mstSet[u] = true;
        for (int v = 0; v < V; v++)
            if (graph[u][v] && !mstSet[v] && graph[u][v] < key[v])
                parent[v] = u, key[v] = graph[u][v];
    }
}`,
    c: `void prims(int graph[][V]) {
    int key[V], parent[V];
    bool mstSet[V];
    for (int i = 0; i < V; i++)
        key[i] = INT_MAX, mstSet[i] = false;
    key[0] = 0;
    parent[0] = -1;
    for (int count = 0; count < V - 1; count++) {
        int u = -1;
        for (int v = 0; v < V; v++)
            if (!mstSet[v] && (u == -1 || key[v] < key[u]))
                u = v;
        mstSet[u] = true;
        for (int v = 0; v < V; v++)
            if (graph[u][v] && !mstSet[v] && graph[u][v] < key[v])
                parent[v] = u, key[v] = graph[u][v];
    }
}`
  },
  'kruskal': {
    python: `def kruskal(graph):
    n = len(graph)
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if graph[i][j] > 0:
                edges.append((graph[i][j], i, j))
    edges.sort()
    parent = [-1] * n
    rank = [0] * n
    def find(u):
        if parent[u] == -1:
            return u
        return find(parent[u])
    def union(u, v):
        rootU = find(u)
        rootV = find(v)
        if rootU != rootV:
            if rank[rootU] > rank[rootV]:
                parent[rootV] = rootU
            elif rank[rootU] < rank[rootV]:
                parent[rootU] = rootV
            else:
                parent[rootV] = rootU
                rank[rootU] += 1
            return True
        return False
    mst = []
    for weight, u, v in edges:
        if union(u, v):
            mst.append((u, v, weight))
    return mst`,
    cpp: `void kruskal(int graph[][V]) {
    vector<pair<int, pair<int, int>>> edges;
    for (int i = 0; i < V; i++) {
        for (int j = i + 1; j < V; j++) {
            if (graph[i][j] > 0) {
                edges.push_back({graph[i][j], {i, j}});
            }
        }
    }
    sort(edges.begin(), edges.end());
    vector<int> parent(V, -1), rank(V, 0);
    vector<pair<int, int>> mst;
    for (auto& edge : edges) {
        int u = edge.second.first;
        int v = edge.second.second;
        if (find(u) != find(v)) {
            union(u, v);
            mst.push_back({u, v});
        }
    }
}`,
    c: `void kruskal(int graph[][V]) {
    vector<pair<int, pair<int, int>>> edges;
    for (int i = 0; i < V; i++) {
        for (int j = i + 1; j < V; j++) {
            if (graph[i][j] > 0) {
                edges.push_back({graph[i][j], {i, j}});
            }
        }
    }
    sort(edges.begin(), edges.end());
    vector<int> parent(V, -1), rank(V, 0);
    vector<pair<int, int>> mst;
    for (auto& edge : edges) {
        int u = edge.second.first;
        int v = edge.second.second;
        if (find(u) != find(v)) {
            union(u, v);
            mst.push_back({u, v});
        }
    }
}`
  },
  'dijkstra': {
    python: `def dijkstra(graph, source):
    n = len(graph)
    dist = [float('inf')] * n
    dist[source] = 0
    pq = [(0, source)]
    while pq:
        current_dist, u = heapq.heappop(pq)
        if current_dist > dist[u]:
            continue
        for v, weight in graph[u]:
            if dist[v] > current_dist + weight:
                dist[v] = current_dist + weight
                heapq.heappush(pq, (dist[v], v))
    return dist`,
    cpp: `void dijkstra(int graph[][V], int source) {
    int dist[V];
    bool sptSet[V];
    for (int i = 0; i < V; i++)
        dist[i] = INT_MAX, sptSet[i] = false;
    dist[source] = 0;
    for (int count = 0; count < V - 1; count++) {
        int u = -1;
        for (int v = 0; v < V; v++)
            if (!sptSet[v] && (u == -1 || dist[v] < dist[u]))
                u = v;
        sptSet[u] = true;
        for (int v = 0; v < V; v++)
            if (!sptSet[v] && graph[u][v] && dist[u] != INT_MAX && dist[u] + graph[u][v] < dist[v])
                dist[v] = dist[u] + graph[u][v];
    }
}`,
    c: `void dijkstra(int graph[][V], int source) {
    int dist[V];
    bool sptSet[V];
    for (int i = 0; i < V; i++)
        dist[i] = INT_MAX, sptSet[i] = false;
    dist[source] = 0;
    for (int count = 0; count < V - 1; count++) {
        int u = -1;
        for (int v = 0; v < V; v++)
            if (!sptSet[v] && (u == -1 || dist[v] < dist[u]))
                u = v;
        sptSet[u] = true;
        for (int v = 0; v < V; v++)
            if (!sptSet[v] && graph[u][v] && dist[u] != INT_MAX && dist[u] + graph[u][v] < dist[v])
                dist[v] = dist[u] + graph[u][v];
    }
}`
  }
};

function MergeSortTree({ node, level = 0 }) {
  if (!node) return null;
  const isLeaf = !node.left && !node.right;
  const boxStyle = {
    display: 'inline-block',
    padding: '8px 14px',
    borderRadius: 8,
    background: isLeaf ? '#e0f7fa' : level === 0 ? '#ffe082' : '#c5e1a5',
    border: '2px solid #232f3e',
    fontWeight: 600,
    color: '#232f3e',
    minWidth: 60,
    textAlign: 'center',
    boxShadow: '0 2px 8px #0001',
    margin: '0 auto',
    position: 'relative',
    zIndex: 1
  };
  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', position: 'relative', marginBottom: 24 }}>
      <div style={boxStyle}>
        <div>Range: [{node.range[0]}, {node.range[1]}]</div>
        <div>Array: [{node.array.join(', ')}]</div>
        {node.merged && (
          <div style={{ color: '#388e3c', fontWeight: 700, marginTop: 4, fontSize: '0.95em' }}>
            Merged: [{node.merged.join(', ')}]
          </div>
        )}
      </div>
      {(node.left || node.right) && (
        <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'flex-start', marginTop: 8, position: 'relative', width: '100%' }}>
          {/* SVG lines connecting parent to children */}
          <svg width="100%" height="24" style={{ position: 'absolute', top: 0, left: 0, zIndex: 0, pointerEvents: 'none' }}>
            {node.left && (
              <line x1="50%" y1="0" x2="25%" y2="24" stroke="#232f3e" strokeWidth="2" />
            )}
            {node.right && (
              <line x1="50%" y1="0" x2="75%" y2="24" stroke="#232f3e" strokeWidth="2" />
            )}
          </svg>
          <div style={{ display: 'flex', gap: 32, width: '100%', justifyContent: 'space-between', zIndex: 1 }}>
            {node.left && <MergeSortTree node={node.left} level={level + 1} />}
            {node.right && <MergeSortTree node={node.right} level={level + 1} />}
          </div>
        </div>
      )}
    </div>
  );
}

function MatrixTable({ matrix }) {
  if (!Array.isArray(matrix) || !matrix.length) return null;
  // Normalize to 2D for rendering: if 1D, wrap as a single row
  const normalized = Array.isArray(matrix[0]) ? matrix : [matrix];
  return (
    <table style={{ borderCollapse: 'collapse', margin: '16px auto', background: '#f8fafc', boxShadow: '0 2px 8px #0001' }}>
      <tbody>
        {normalized.map((row, i) => (
          <tr key={i}>
            {row.map((cell, j) => (
              <td
                key={j}
                style={{
                  border: '1px solid #bbb',
                  padding: '8px 14px',
                  background: cell === 0 ? '#fff' : cell === 1 ? '#b2dfdb' : (typeof cell === 'number' && !isFinite(cell)) ? '#eee' : '#ffe082',
                  color: (typeof cell === 'number' && !isFinite(cell)) ? '#888' : '#232f3e',
                  fontWeight: 600,
                  minWidth: 32,
                  textAlign: 'center'
                }}
              >
                {typeof cell === 'number' && !isFinite(cell) ? '∞' : Array.isArray(cell) ? JSON.stringify(cell) : cell}
              </td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function TopoGraph({ matrix, order }) {
  if (!Array.isArray(matrix) || !matrix.length) return null;
  const n = matrix.length;
  // Node positions (simple horizontal layout)
  const nodeRadius = 22;
  const nodeGap = 80;
  const width = n * nodeGap + 40;
  const height = 120;
  const nodeY = 60;
  // Map topological order to highlight
  return (
    <svg width={width} height={height} style={{ display: 'block', margin: '24px auto', background: '#f8fafc', borderRadius: 12, boxShadow: '0 2px 8px #0001' }}>
      {/* Edges */}
      {matrix.map((row, i) => row.map((val, j) => val ? (
        <g key={`edge-${i}-${j}`}>
          <line
            x1={20 + i * nodeGap + nodeRadius}
            y1={nodeY}
            x2={20 + j * nodeGap + nodeRadius}
            y2={nodeY}
            stroke="#888"
            strokeWidth={2}
            markerEnd="url(#arrowhead)"
          />
        </g>
      ) : null))}
      {/* Arrowhead marker */}
      <defs>
        <marker id="arrowhead" markerWidth="8" markerHeight="8" refX="8" refY="4" orient="auto" markerUnits="strokeWidth">
          <path d="M0,0 L8,4 L0,8" fill="#888" />
        </marker>
      </defs>
      {/* Nodes */}
      {Array.from({ length: n }).map((_, i) => (
        <g key={`node-${i}`}>
          <circle
            cx={20 + i * nodeGap + nodeRadius}
            cy={nodeY}
            r={nodeRadius}
            fill={order && order.includes(i) ? '#ffe082' : '#b2dfdb'}
            stroke="#232f3e"
            strokeWidth={2}
          />
          <text
            x={20 + i * nodeGap + nodeRadius}
            y={nodeY + 6}
            textAnchor="middle"
            fontWeight={700}
            fontSize={18}
            fill="#232f3e"
          >
            {i}
          </text>
        </g>
      ))}
      {/* Topological order labels */}
      {order && order.length > 0 && (
        <text x={width / 2} y={height - 10} textAnchor="middle" fontSize={16} fill="#388e3c" fontWeight={600}>
          Topological Order: {order.join(' → ')}
        </text>
      )}
    </svg>
  );
}

// Add diagram for Selection Sort
function SelectionSortDiagram({ steps, stepIndex }) {
  // Parse the step to extract array and highlight info
  // We'll look for lines like: 'Step X: Start from index i, current array: [...]', 'Compare arr[j]=x with current min arr[min_idx]=y', 'Swap arr[i] and arr[min_idx]: [...]'
  let arr = [];
  let current = -1, minIdx = -1, swapIdx = -1, swapWith = -1;
  for (let i = stepIndex; i >= 0; i--) {
    const s = steps[i];
    const arrMatch = s.match(/current array: (\[.*\])/);
    if (arrMatch) {
      try { arr = JSON.parse(arrMatch[1]); break; } catch {}
    }
    const swapMatch = s.match(/Swap arr\[(\d+)\] and arr\[(\d+)\]: (\[.*\])/);
    if (swapMatch) {
      try { arr = JSON.parse(swapMatch[3]); swapIdx = parseInt(swapMatch[1]); swapWith = parseInt(swapMatch[2]); break; } catch {}
    }
  }
  for (let i = stepIndex; i >= 0; i--) {
    const s = steps[i];
    const minMatch = s.match(/min at index (\d+)/) || s.match(/min found at index (\d+)/) || s.match(/min_idx\]=(\d+)/);
    if (minMatch) { minIdx = parseInt(minMatch[1]); break; }
    const minMatch2 = s.match(/current min arr\[(\d+)\]=/);
    if (minMatch2) { minIdx = parseInt(minMatch2[1]); break; }
  }
  for (let i = stepIndex; i >= 0; i--) {
    const s = steps[i];
    const curMatch = s.match(/Start from index (\d+)/);
    if (curMatch) { current = parseInt(curMatch[1]); break; }
  }
  return (
    <div style={{ display: 'flex', justifyContent: 'center', gap: 8, marginTop: 12 }}>
      {arr.map((val, idx) => (
        <div key={idx} style={{
          padding: '12px 16px',
          borderRadius: 8,
          background: idx === swapIdx || idx === swapWith ? '#ffe082' : idx === minIdx ? '#b2dfdb' : idx === current ? '#90caf9' : '#f3f4f6',
          border: '2px solid #232f3e',
          fontWeight: 700,
          color: '#232f3e',
          minWidth: 32,
          textAlign: 'center',
          fontSize: 18,
          boxShadow: idx === swapIdx || idx === swapWith ? '0 2px 8px #ff0a' : undefined
        }}>{val}</div>
      ))}
    </div>
  );
}

// Add diagram for Quick Sort
function QuickSortDiagram({ steps, stepIndex }) {
  // Robustly parse the array for the current step or fallback to last known
  let arr = [];
  let pivot = null, pivotIdx = -1, compareIdx = -1, swapIdx = -1, swapWith = -1;
  // Find the most recent array state up to this step
  for (let i = stepIndex; i >= 0; i--) {
    const s = steps[i];
    const arrMatch = s.match(/Partitioning: (\[.*\]), pivot=(-?\d+)/);
    if (arrMatch) {
      try { arr = JSON.parse(arrMatch[1]); pivot = parseInt(arrMatch[2]); break; } catch {}
    }
    const swapMatch = s.match(/Swap a\[(\d+)\] and a\[(\d+)\]: (\[.*\])/);
    if (swapMatch) {
      try { arr = JSON.parse(swapMatch[3]); swapIdx = parseInt(swapMatch[1]); swapWith = parseInt(swapMatch[2]); break; } catch {}
    }
    const arrOnlyMatch = s.match(/: (\[.*\])$/); // fallback for any array at end
    if (arrOnlyMatch) {
      try { arr = JSON.parse(arrOnlyMatch[1]); break; } catch {}
    }
  }
  // If still empty, try to find any array in the steps
  if (!arr.length) {
    for (let i = steps.length - 1; i >= 0; i--) {
      const arrMatch = steps[i].match(/(\[.*\])/);
      if (arrMatch) {
        try { arr = JSON.parse(arrMatch[1]); break; } catch {}
      }
    }
  }
  // Find compare index
  for (let i = stepIndex; i >= 0; i--) {
    const s = steps[i];
    const compMatch = s.match(/Compare a\[(\d+)\]=/);
    if (compMatch) { compareIdx = parseInt(compMatch[1]); break; }
  }
  // Find pivot index
  if (pivot === null && arr.length) {
    // Try to find a line with 'pivot=' in it
    for (let i = stepIndex; i >= 0; i--) {
      const s = steps[i];
      const pivotMatch = s.match(/pivot=(-?\d+)/);
      if (pivotMatch) { pivot = parseInt(pivotMatch[1]); break; }
    }
  }
  if (pivot !== null && arr.length) {
    for (let i = arr.length - 1; i >= 0; i--) {
      if (arr[i] === pivot) { pivotIdx = i; break; }
    }
  }
  return (
    <div style={{ display: 'flex', justifyContent: 'center', gap: 8, marginTop: 12 }}>
      {arr.map((val, idx) => (
        <div key={idx} style={{
          padding: '12px 16px',
          borderRadius: 8,
          background: idx === swapIdx || idx === swapWith ? '#ffe082' : idx === pivotIdx ? '#b2dfdb' : idx === compareIdx ? '#90caf9' : '#f3f4f6',
          border: '2px solid #232f3e',
          fontWeight: 700,
          color: '#232f3e',
          minWidth: 32,
          textAlign: 'center',
          fontSize: 18,
          boxShadow: idx === swapIdx || idx === swapWith ? '0 2px 8px #ff0a' : undefined
        }}>{val}</div>
      ))}
    </div>
  );
}

function IntroCarousel() {
  const cards = [
    {
      title: 'Welcome to Algorithm Analyzer!',
      text: 'A modern tool to visualize, analyze, and learn classic algorithms with interactive diagrams and step-by-step explanations.'
    },
    {
      title: 'Visualize Algorithms',
      text: 'See how Merge Sort, Quick Sort, and more work under the hood with real-time diagrams and trees.'
    },
    {
      title: 'Step-by-Step Analysis',
      text: 'Get detailed, human-readable explanations for every step of the algorithm.'
    },
    {
      title: 'Try It Yourself!',
      text: 'Select an algorithm, enter your own input, and click Visualize to get started.'
    }
  ];
  const [idx, setIdx] = React.useState(0);
  React.useEffect(() => {
    const timer = setTimeout(() => setIdx((idx + 1) % cards.length), 3500);
    return () => clearTimeout(timer);
  }, [idx, cards.length]);
  return (
    <div style={{
      maxWidth: 600,
      margin: '0 auto 2rem auto',
      padding: 0,
      position: 'relative',
      minHeight: 110
    }}>
      <div style={{
        background: '#fff',
        borderRadius: 14,
        boxShadow: '0 4px 24px #0001',
        border: '2px solid #43a047',
        padding: '1.5rem 2.5rem',
        textAlign: 'center',
        transition: 'all 0.5s',
        minHeight: 90
      }}>
        <div style={{ fontWeight: 800, fontSize: '1.35rem', color: '#232f3e', marginBottom: 8 }}>{cards[idx].title}</div>
        <div style={{ fontSize: '1.08rem', color: '#388e3c', fontWeight: 500 }}>{cards[idx].text}</div>
      </div>
      <div style={{ position: 'absolute', bottom: 10, left: 0, right: 0, textAlign: 'center' }}>
        {cards.map((_, i) => (
          <span key={i} style={{
            display: 'inline-block',
            width: 10,
            height: 10,
            borderRadius: '50%',
            background: i === idx ? '#43a047' : '#c8e6c9',
            margin: '0 4px',
            transition: 'background 0.3s'
          }} />
        ))}
      </div>
    </div>
  );
}

function App() {
  // Remove page and setPage, only show the visualizer
  const [algorithm, setAlgorithm] = useState(ALGORITHMS[0].value);
  const [input, setInput] = useState('');
  const [output, setOutput] = useState('');
  const [treeData, setTreeData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [steps, setSteps] = useState([]);
  const [topoOrder, setTopoOrder] = useState([]);
  const [showInfo, setShowInfo] = useState(false);
  const [pivotStrategy, setPivotStrategy] = useState('last');
  const [customPivot, setCustomPivot] = useState('');
  const [infoLang, setInfoLang] = useState('python');
  const [stepIndex, setStepIndex] = useState(0);
  const [knapsackWeights, setKnapsackWeights] = useState('');
  const [knapsackProfits, setKnapsackProfits] = useState('');
  const [knapsackCapacity, setKnapsackCapacity] = useState('');
  const [dijkstraSource, setDijkstraSource] = useState('0');
  const [matrices, setMatrices] = useState([]);
  const [matrixStep, setMatrixStep] = useState(0);

  // When steps change, reset stepIndex
  React.useEffect(() => { setStepIndex(0); }, [steps]);

  const handleVisualize = async () => {
    setLoading(true);
    setOutput('');
    setTreeData(null);
    setSteps([]);
    let arr;
    let endpoint = '';
    if (algorithm === 'merge-sort' || algorithm === 'quick-sort' || algorithm === 'selection-sort') {
      try {
        arr = input.split(/,|\s+/).map(x => parseInt(x)).filter(x => !isNaN(x));
        if (!arr.length) throw new Error('No valid numbers');
      } catch {
        setOutput('Invalid input. Please enter numbers separated by commas or spaces.');
        setLoading(false);
        return;
      }
    }
    if (algorithm === 'merge-sort') {
      endpoint = 'merge-sort';
    } else if (algorithm === 'quick-sort') {
      endpoint = 'quick-sort';
    } else if (algorithm === 'selection-sort') {
      endpoint = 'selection-sort';
    } else if (algorithm === 'topo-sort') {
      endpoint = 'topo-sort';
    } else if (algorithm === 'floyd-warshall') {
      endpoint = 'floyd-warshall';
    } else if (algorithm === 'warshall') {
      endpoint = 'warshall';
    } else if (algorithm === 'activity-selection') {
      endpoint = 'activity-selection';
    } else if (algorithm === 'knapsack') {
      endpoint = 'knapsack';
    } else if (algorithm === 'prims') {
      endpoint = 'prims';
    } else if (algorithm === 'kruskal') {
      endpoint = 'kruskal';
    } else if (algorithm === 'dijkstra') {
      endpoint = 'dijkstra';
    }
    if (endpoint) {
      try {
        let body;
        if (algorithm === 'topo-sort' || algorithm === 'floyd-warshall' || algorithm === 'warshall' || algorithm === 'prims' || algorithm === 'kruskal' || algorithm === 'dijkstra') {
          let matrix;
          try {
            matrix = JSON.parse(input);
            if (!Array.isArray(matrix) || !matrix.every(row => Array.isArray(row) && row.length === matrix.length)) throw new Error();
          } catch {
            setOutput('Invalid input. Please enter a valid square adjacency matrix as a JSON array.');
            setLoading(false);
            return;
          }
          body = JSON.stringify({ matrix, ...(algorithm === 'dijkstra' ? { source: parseInt(dijkstraSource) } : {}) });
        } else if (algorithm === 'activity-selection') {
          let activities;
          try {
            activities = JSON.parse(input);
            if (!Array.isArray(activities) || !activities.every(x => Array.isArray(x) && x.length === 2)) throw new Error();
          } catch {
            setOutput('Invalid input. Please enter a list of [start, end] pairs as a JSON array.');
            setLoading(false);
            return;
          }
          body = JSON.stringify({ activities });
        } else if (algorithm === 'knapsack') {
          let weights, profits, capacity;
          try {
            weights = knapsackWeights.split(/,|\s+/).map(x => parseInt(x)).filter(x => !isNaN(x));
            profits = knapsackProfits.split(/,|\s+/).map(x => parseInt(x)).filter(x => !isNaN(x));
            capacity = parseInt(knapsackCapacity);
            if (!weights.length || !profits.length || isNaN(capacity) || weights.length !== profits.length) throw new Error();
          } catch {
            setOutput('Invalid input. Enter weights and profits as comma/space separated numbers, and capacity as an integer.');
            setLoading(false);
            return;
          }
          body = JSON.stringify({ weights, profits, capacity });
        } else if (algorithm === 'quick-sort') {
          body = JSON.stringify({
            array: arr,
            pivot_strategy: pivotStrategy,
            ...(pivotStrategy === 'custom' && customPivot !== '' ? { pivot_index: parseInt(customPivot) } : {})
          });
        } else {
          body = JSON.stringify({ array: arr });
        }
        const res = await fetch(`${API_BASE}/api/${endpoint}`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body
        });
        const data = await res.json();
        if (data.result !== undefined || data.edges || data.distances) {
          setOutput(
            algorithm === 'topo-sort'
              ? 'Topological Order: ' + JSON.stringify(data.result)
              : algorithm === 'floyd-warshall'
                ? 'Shortest Path Matrix: ' + JSON.stringify(data.result)
              : algorithm === 'warshall'
                ? 'Transitive Closure: ' + JSON.stringify(data.result)
              : algorithm === 'activity-selection'
                ? 'Selected Activities: ' + JSON.stringify(data.result)
              : algorithm === 'knapsack'
                ? `Max Profit: ${data.result}, Items: ${JSON.stringify(data.items)}`
              : algorithm === 'prims' || algorithm === 'kruskal'
                ? `MST Edges: ${JSON.stringify(data.edges)}, Total Cost: ${data.total}`
              : algorithm === 'dijkstra'
                ? `Distances: ${JSON.stringify(data.distances)}, Paths: ${JSON.stringify(data.paths)}`
              : 'Sorted: ' + JSON.stringify(data.result)
          );
          setSteps(data.steps || []);
          setTreeData(
            algorithm === 'merge-sort' ? data.tree :
            algorithm === 'topo-sort' ? data.matrix :
            algorithm === 'floyd-warshall' || algorithm === 'warshall' ? data.result :
            algorithm === 'knapsack' ? data.matrix :
            null
          );
          if (algorithm === 'topo-sort') setTopoOrder(data.result);
          if (algorithm === 'floyd-warshall' || algorithm === 'warshall') {
            setMatrices(data.matrices || []);
            setMatrixStep(0);
          } else {
            setMatrices([]);
            setMatrixStep(0);
          }
        } else {
          setOutput('Error: ' + (data.error || 'Unknown error'));
        }
      } catch (e) {
        setOutput('Error connecting to backend. Is it running?');
      }
    }
    setLoading(false);
  };

  return (
    <div className="App" style={{ background: '#f3f4f6', minHeight: '100vh' }}>
      <header style={{
        background: '#232f3e',
        color: 'white',
        padding: '1rem 2rem',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        boxShadow: '0 4px 16px #0002',
        borderBottom: '4px solid #232f3e',
        position: 'sticky',
        top: 0,
        zIndex: 100
      }}>
        <h1 style={{ margin: 0, fontWeight: 900, fontSize: '2.5rem', letterSpacing: '2px', color: '#fff', textShadow: '0 2px 8px #232f3e55' }}>Algorithm Analyzer</h1>
      </header>
      <main style={{ maxWidth: 900, margin: '2rem auto', padding: '2rem', background: 'white', borderRadius: 16, boxShadow: '0 4px 24px #0001' }}>
        <IntroCarousel />
        <h2 style={{ color: '#232f3e' }}>Algorithm Visualizer</h2>
        <div style={{ margin: '2rem 0', padding: '1rem', background: '#f9fafb', borderRadius: 8, border: '1px solid #e5e7eb' }}>
          <label style={{ fontWeight: 600, color: '#232f3e' }}>Select Algorithm:</label>
          <select value={algorithm} onChange={e => setAlgorithm(e.target.value)} style={{ marginLeft: 12, padding: 6, borderRadius: 4 }}>
            {ALGORITHMS.map(a => <option key={a.value} value={a.value}>{a.label}</option>)}
          </select>
          <button
            style={{ marginLeft: 16, padding: '4px 12px', background: '#e0e7ef', color: '#232f3e', border: '1px solid #bbb', borderRadius: 4, fontWeight: 600, cursor: 'pointer' }}
            onClick={() => setShowInfo(true)}
            title="Show algorithm info"
          >
            ℹ️ Info
          </button>
          {algorithm === 'quick-sort' && (
            <span style={{ marginLeft: 24 }}>
              <label style={{ fontWeight: 600, color: '#232f3e', marginRight: 8 }}>Pivot:</label>
              <select value={pivotStrategy} onChange={e => setPivotStrategy(e.target.value)} style={{ padding: 4, borderRadius: 4 }}>
                <option value="first">First Element</option>
                <option value="last">Last Element</option>
                <option value="random">Random</option>
                <option value="custom">Custom Index</option>
              </select>
              {pivotStrategy === 'custom' && (
                <input
                  type="number"
                  min="0"
                  value={customPivot}
                  onChange={e => setCustomPivot(e.target.value)}
                  placeholder="Index"
                  style={{ marginLeft: 8, width: 60, padding: 4, borderRadius: 4 }}
                />
              )}
            </span>
          )}
          <div style={{ marginTop: 20 }}>
            <label style={{ fontWeight: 600, color: '#232f3e' }}>Input Data:</label>
            {/* Only show main input for these algorithms */}
            {(algorithm === 'merge-sort' || algorithm === 'quick-sort' || algorithm === 'selection-sort' || algorithm === 'topo-sort' || algorithm === 'floyd-warshall' || algorithm === 'warshall' || algorithm === 'activity-selection') && (
              <input
                type="text"
                value={input}
                onChange={e => setInput(e.target.value)}
                placeholder={
                  algorithm === 'topo-sort'
                    ? 'e.g. [[0,1,0],[0,0,1],[0,0,0]] (1=edge, 0=no edge)'
                    : algorithm === 'floyd-warshall'
                      ? 'e.g. [[0,3,0,5],[2,0,0,4],[0,1,0,0],[0,0,2,0]] (0=no edge, positive=weight)'
                      : algorithm === 'warshall'
                        ? 'e.g. [[0,1,0],[0,0,1],[0,0,0]] (0/1 only)'
                        : algorithm === 'activity-selection'
                          ? 'e.g. [ [1,2], [3,4], [0,6], [5,7], [8,9], [5,9] ]'
                          : 'e.g. 5, 2, 9, 1, 6'
                }
                style={{ marginLeft: 12, padding: 6, borderRadius: 4, width: 300 }}
              />
            )}
            {/* Knapsack fields */}
            {algorithm === 'knapsack' && (
              <span>
                <label style={{ fontWeight: 600, color: '#232f3e', marginLeft: 8 }}>Weights:</label>
                <input type="text" value={knapsackWeights} onChange={e => setKnapsackWeights(e.target.value)} placeholder="e.g. 2, 3, 4, 5" style={{ marginLeft: 8, padding: 6, borderRadius: 4, width: 120 }} />
                <label style={{ fontWeight: 600, color: '#232f3e', marginLeft: 8 }}>Profits:</label>
                <input type="text" value={knapsackProfits} onChange={e => setKnapsackProfits(e.target.value)} placeholder="e.g. 3, 4, 5, 6" style={{ marginLeft: 8, padding: 6, borderRadius: 4, width: 120 }} />
                <label style={{ fontWeight: 600, color: '#232f3e', marginLeft: 8 }}>Capacity:</label>
                <input type="number" value={knapsackCapacity} onChange={e => setKnapsackCapacity(e.target.value)} placeholder="e.g. 5" style={{ marginLeft: 8, padding: 6, borderRadius: 4, width: 80 }} />
              </span>
            )}
            {/* Graph algorithm fields */}
            {(algorithm === 'prims' || algorithm === 'kruskal' || algorithm === 'dijkstra') && (
              <span>
                <label style={{ fontWeight: 600, color: '#232f3e', marginLeft: 8 }}>Adjacency Matrix:</label>
                <input type="text" value={input} onChange={e => setInput(e.target.value)} placeholder="e.g. [[0,2,0],[2,0,3],[0,3,0]] (0 or inf = no edge)" style={{ marginLeft: 8, padding: 6, borderRadius: 4, width: 300 }} />
                {algorithm === 'dijkstra' && (
                  <span style={{ marginLeft: 16 }}>
                    <label style={{ fontWeight: 600, color: '#232f3e' }}>Source:</label>
                    <input type="number" value={dijkstraSource} onChange={e => setDijkstraSource(e.target.value)} min="0" style={{ marginLeft: 8, width: 60, padding: 4, borderRadius: 4 }} />
                  </span>
                )}
              </span>
            )}
          </div>
          {/* Add Visualize button below input fields for all algorithms */}
          <div style={{ marginTop: 24, textAlign: 'center' }}>
            <button
              onClick={handleVisualize}
              style={{
                padding: '10px 32px',
                background: '#388e3c',
                color: 'white',
                border: 'none',
                borderRadius: 6,
                fontWeight: 700,
                fontSize: '1.1rem',
                cursor: 'pointer',
                boxShadow: '0 2px 8px #0002',
                opacity: loading ? 0.7 : 1
              }}
              disabled={loading}
            >
              {loading ? 'Visualizing...' : 'Visualize'}
            </button>
          </div>
          <div style={{ marginTop: 30 }}>
            <label style={{ fontWeight: 600, color: '#232f3e' }}>Output:</label>
            <div style={{ marginTop: 8, background: '#fff', border: '1px solid #e5e7eb', borderRadius: 4, padding: 12, minHeight: 40 }}>
              {output}
            </div>
            {steps.length > 0 && (
              <div style={{ marginTop: 20 }}>
                <label style={{ fontWeight: 600, color: '#232f3e' }}>Step-by-step Explanation:</label>
                <ol style={{ marginTop: 8, background: '#f8fafc', border: '1px solid #e5e7eb', borderRadius: 4, padding: 16 }}>
                  {steps.map((step, idx) => <li key={idx} style={{ marginBottom: 6 }}>{step}</li>)}
                </ol>
              </div>
            )}
            {/* Intermediate matrices stepper for Floyd-Warshall and Warshall */}
            {(algorithm === 'floyd-warshall' || algorithm === 'warshall') && matrices.length > 0 && (
              <div style={{ marginTop: 24, textAlign: 'center' }}>
                <label style={{ fontWeight: 600, color: '#232f3e' }}>Intermediate Matrix (Step {matrixStep} of {matrices.length - 1}):</label>
                <MatrixTable matrix={matrices[matrixStep]} />
                <div style={{ marginTop: 12 }}>
                  <button onClick={() => setMatrixStep(s => Math.max(0, s - 1))} disabled={matrixStep === 0} style={{ marginRight: 12, padding: '6px 18px', borderRadius: 4, border: '1px solid #bbb', background: '#e0e7ef', color: '#232f3e', fontWeight: 600, cursor: matrixStep === 0 ? 'not-allowed' : 'pointer' }}>Prev</button>
                  <button onClick={() => setMatrixStep(s => Math.min(matrices.length - 1, s + 1))} disabled={matrixStep === matrices.length - 1} style={{ padding: '6px 18px', borderRadius: 4, border: '1px solid #bbb', background: '#e0e7ef', color: '#232f3e', fontWeight: 600, cursor: matrixStep === matrices.length - 1 ? 'not-allowed' : 'pointer' }}>Next</button>
                </div>
              </div>
            )}
            {(algorithm === 'floyd-warshall' || algorithm === 'warshall' || algorithm === 'prims' || algorithm === 'kruskal' || algorithm === 'dijkstra') && treeData && Array.isArray(treeData) && (
              <div style={{ marginTop: 24 }}>
                <label style={{ fontWeight: 600, color: '#232f3e' }}>Matrix Visualization:</label>
                <MatrixTable matrix={treeData} />
              </div>
            )}
            {(algorithm === 'floyd-warshall' || algorithm === 'warshall' || algorithm === 'prims' || algorithm === 'kruskal' || algorithm === 'dijkstra') && !treeData && output && output.includes('[') && (
              <div style={{ marginTop: 24 }}>
                <label style={{ fontWeight: 600, color: '#232f3e' }}>Matrix Visualization:</label>
                <MatrixTable matrix={(() => { try { return JSON.parse(output.match(/\[.*\]/s)[0]); } catch { return null; } })()} />
              </div>
            )}
            {algorithm === 'topo-sort' && treeData && Array.isArray(treeData) && (
              <div style={{ marginTop: 24 }}>
                <label style={{ fontWeight: 600, color: '#232f3e' }}>Graph Visualization:</label>
                <TopoGraph matrix={treeData} order={topoOrder} />
              </div>
            )}
            {algorithm === 'selection-sort' && steps.length > 0 && (
              <SelectionSortDiagram steps={steps} stepIndex={stepIndex} />
            )}
            {algorithm === 'quick-sort' && steps.length > 0 && (
              <QuickSortDiagram steps={steps} stepIndex={stepIndex} />
            )}
          </div>
          {algorithm === 'knapsack' && (
            <div style={{ marginTop: 16 }}>
              <label style={{ fontWeight: 600, color: '#232f3e' }}>Weights:</label>
              <input type="text" value={knapsackWeights} onChange={e => setKnapsackWeights(e.target.value)} placeholder="e.g. 2, 3, 4, 5" style={{ marginLeft: 8, padding: 6, borderRadius: 4, width: 180 }} />
              <label style={{ fontWeight: 600, color: '#232f3e', marginLeft: 16 }}>Profits:</label>
              <input type="text" value={knapsackProfits} onChange={e => setKnapsackProfits(e.target.value)} placeholder="e.g. 3, 4, 5, 6" style={{ marginLeft: 8, padding: 6, borderRadius: 4, width: 180 }} />
              <label style={{ fontWeight: 600, color: '#232f3e', marginLeft: 16 }}>Capacity:</label>
              <input type="number" value={knapsackCapacity} onChange={e => setKnapsackCapacity(e.target.value)} placeholder="e.g. 5" style={{ marginLeft: 8, padding: 6, borderRadius: 4, width: 80 }} />
            </div>
          )}
          {(algorithm === 'prims' || algorithm === 'kruskal' || algorithm === 'dijkstra') && (
            <div style={{ marginTop: 16 }}>
              <label style={{ fontWeight: 600, color: '#232f3e' }}>Adjacency Matrix:</label>
              <input type="text" value={input} onChange={e => setInput(e.target.value)} placeholder="e.g. [[0,2,0],[2,0,3],[0,3,0]] (0 or inf = no edge)" style={{ marginLeft: 8, padding: 6, borderRadius: 4, width: 400 }} />
              {algorithm === 'dijkstra' && (
                <span style={{ marginLeft: 16 }}>
                  <label style={{ fontWeight: 600, color: '#232f3e' }}>Source:</label>
                  <input type="number" value={dijkstraSource} onChange={e => setDijkstraSource(e.target.value)} min="0" style={{ marginLeft: 8, width: 60, padding: 4, borderRadius: 4 }} />
                </span>
              )}
            </div>
          )}
          {/* Matrix/Table Visualization for Knapsack and Graph Algorithms */}
          {(algorithm === 'knapsack' && treeData && Array.isArray(treeData)) && (
            <div style={{ marginTop: 24 }}>
              <label style={{ fontWeight: 600, color: '#232f3e' }}>DP Matrix:</label>
              <MatrixTable matrix={treeData} />
            </div>
          )}
          {((algorithm === 'prims' || algorithm === 'kruskal' || algorithm === 'dijkstra') && input && input.startsWith('[')) && (
            <div style={{ marginTop: 24 }}>
              <label style={{ fontWeight: 600, color: '#232f3e' }}>Input Matrix:</label>
              <MatrixTable matrix={(() => { try { return JSON.parse(input); } catch { return null; } })()} />
            </div>
          )}
          {treeData && algorithm === 'merge-sort' && (
            <div style={{ marginTop: 30 }}>
              <label style={{ fontWeight: 600, color: '#232f3e' }}>Tree Visualization:</label>
              <div style={{ marginTop: 8, background: '#fff', border: '1px solid #e5e7eb', borderRadius: 4, padding: 12, minHeight: 120, textAlign: 'left', color: '#222', overflowX: 'auto' }}>
                <MergeSortTree node={treeData} />
              </div>
            </div>
          )}
        </div>
        </main>
      {showInfo && (
        <div style={{ position: 'fixed', top: 0, left: 0, width: '100vw', height: '100vh', background: '#0008', zIndex: 1000, display: 'flex', alignItems: 'center', justifyContent: 'center' }} onClick={() => setShowInfo(false)}>
          <div style={{ background: 'white', borderRadius: 12, padding: 32, minWidth: 320, maxWidth: 520, boxShadow: '0 4px 24px #0002', position: 'relative' }} onClick={e => e.stopPropagation()}>
            <button onClick={() => setShowInfo(false)} style={{ position: 'absolute', top: 12, right: 16, background: 'none', border: 'none', fontSize: 22, color: '#888', cursor: 'pointer' }}>&times;</button>
            <h2 style={{ margin: 0, color: '#232f3e', fontWeight: 700 }}>{ALGO_INFO[algorithm].name}</h2>
            <div style={{ margin: '16px 0', fontSize: '1.1rem', color: '#232f3e' }}>{ALGO_INFO[algorithm].desc}</div>
            <div style={{ fontWeight: 600, color: '#388e3c', fontSize: '1.05rem' }}>Time Complexity: {ALGO_INFO[algorithm].time}</div>
            <div style={{ marginTop: 18 }}>
              <label style={{ fontWeight: 600, color: '#232f3e', marginRight: 8 }}>Show code in:</label>
              <select value={infoLang} onChange={e => setInfoLang(e.target.value)} style={{ padding: 4, borderRadius: 4 }}>
                <option value="python">Python</option>
                <option value="cpp">C++</option>
                <option value="c">C</option>
              </select>
            </div>
            <pre style={{ marginTop: 12, background: '#f8fafc', border: '1px solid #e5e7eb', borderRadius: 6, padding: 14, fontSize: '0.98em', maxHeight: 320, overflow: 'auto', color: '#232f3e' }}>
              {ALGO_CODE[algorithm][infoLang]}
            </pre>
          </div>
        </div>
      )}
      <footer style={{ textAlign: 'center', color: '#888', padding: '1rem 0' }}>
        &copy; {new Date().getFullYear()} Smart Utility Suite | ADA Project
      </footer>
    </div>
  );
}

export default App;

