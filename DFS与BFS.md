## DFS
* 图示：
![image](https://github.com/MissAquarius/ForJobHunting/blob/master/image/DFS.png)

* 例题：
  * [301 remove-invalid-parentheses](https://leetcode.com/problems/remove-invalid-parentheses/)（见BFS部分其他解法）
  * [207 Course Schedule](https://leetcode.com/problems/course-schedule/) （见BFS部分其他解法）
  * [200 Number of Islands](https://leetcode.com/problems/number-of-islands/)（见BFS部分其他解法）
  * [101 Symmetric Tree](https://leetcode.com/problems/symmetric-tree/)（见BFS部分其他解法）
## BFS
* 常用于：求最短路径、至少需要几步的问题
* 搜索过程（借助队列和一个一维数组实现）：
 * 首先将第一个节点push进队列
 * 只要队列不为空，就开始遍历：
   * 访问队头元素
   * 如果该队头元素有子节点，将该元素的所有子节点依次加入队尾
   * 如果该队头元素没有子节点，就继续出队访问
   * 如果该队头就是要寻找的节点，就输出
* 图示：
![image](https://github.com/MissAquarius/ForJobHunting/blob/master/image/BFS.jpg)
* 将0 加入队列:[0]
* 访问队列头部元素 0，将0 的孩子加入队列[5,1,2,6]
* 访问头部元素5，将5的孩子加入队列[1,2,6,3,4]
* 访问头部元素1，1没有孩子，直接continue
* 同理访问2/6/3/4
* 最后遍历顺序为：0->5->1->2->6->3->4
* 伪代码如下：
 ```
 Q.push(head)
 while Q is not empty:
   temp = Q.front
   Q.pop()
   if temp == target:
     break;
     print()
   if temp.hasleftChild or temp.hasrightChild:
      Q.push(temp.leftChild)
      Q.push(temp.rightChild)
   else:
      continue
 ```
## leetcode 例题
* [994. Rotting Oranges](https://leetcode.com/problems/rotting-oranges/)
```python
class Solution(object):
    def orangesRotting(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: int
        """

        m = len(grid)
        n = len(grid[0])
        fresh = 0
        q = collections.deque()
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 1:
                    fresh += 1
                if grid[i][j] == 2:
                    q.append([i, j])

        if fresh == 0:
            return 0
        step = 0
        while q:
            size = len(q)
            for i in range(size):
                x, y = q.popleft()
                # 左边
                if y - 1 >= 0 and grid[x][y - 1] == 1:
                    grid[x][y - 1] = 2
                    fresh -= 1
                    q.append([x, y - 1])

                # 右边
                if y + 1 < n and grid[x][y + 1] == 1:
                    grid[x][y + 1] = 2
                    fresh -= 1
                    q.append([x, y + 1])

                # 上边
                if x - 1 >= 0 and grid[x - 1][y] == 1:
                    grid[x - 1][y] = 2
                    fresh -= 1
                    q.append([x - 1, y])

                # 下边
                if x + 1 < m and grid[x + 1][y] == 1:
                    grid[x + 1][y] = 2
                    fresh -= 1
                    q.append([x + 1, y])
            step += 1

        if fresh != 0:
            return -1
            
        # 注意：为什么要减去1呢？因为while结束时，q已经为空了，说明在上一步没有任何一个橘子被传染，上一步就是终点步了，而step又多加了1，因此要减去
        return step - 1  
        
```
题目描述：这道题类似华为的笔试题，输入一个二维矩阵，取值为1代表新鲜的橘子，为2表示坏橘子，为0表示空。每一分钟，坏橘子会传染其四个方向的新鲜橘子，求所有的橘子被传染需要多久<br>
思路：采用BFS的思想，初始时，统计好橘子的个数fresh，并且把怀橘子的坐标对加入队列中。每一分钟，如果队列不为空，就遍历队列中的所有元素，对于每一个元素，去传染其四个方向的元素，并把传染后的元素坐标对入队，待下一分钟继续遍历。如果队列为空，表示当前能传染其他橘子的橘子都已经传染了，此时检查fresh，如果fresh==0，返回时间，否则，返回-1（表示无法传染所有新鲜橘子）.

* [993 Cousins in Binary Tree](https://leetcode.com/problems/cousins-in-binary-tree/)
```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def isCousins(self, root, x, y):
        """
        :type root: TreeNode
        :type x: int
        :type y: int
        :rtype: bool
        """
        dict1 = {} # 存储每个node的值及父节点和层次
        q = collections.deque()
        depth = 0
        
        # 根节点先入队
        q.append([root, None])
        while q:
            size = len(q)
            for i in range(size):
                node, parent = q.popleft()
                if node == None:
                    continue
    
                dict1[node.val] =[parent, depth]
                q.append([node.left, node])
                q.append([node.right, node])
            depth += 1
            
        x_parent = dict1[x][0]
        x_depth = dict1[x][1] 
        y_parent = dict1[y][0]
        y_depth = dict1[y][1]
        
        return x_depth == y_depth and x_parent != y_parent
```
解法一：BFS思路 首先用BFS方法遍历整棵树，并把每个节点对应的parent和depth放入到一个字典中；最后判断（由于要记录父节点，因为队列中不仅存放了当前节点，还存放了其父节点）

```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def dfs(self, root, parent, depth):
        if root == None:
            return 
        self.dict1[root.val] = [parent,depth]
        self.dfs(root.left, root,depth + 1)
        self.dfs(root.right,root,depth + 1)
    
    def isCousins(self, root, x, y):
        """
        :type root: TreeNode
        :type x: int
        :type y: int
        :rtype: bool
        """
        self.dict1 = {} # 存储每个node的值及父节点和层次
        self.dfs(root, None, 0) 
        x_parent = self.dict1[x][0]
        x_depth = self.dict1[x][1] 
        y_parent = self.dict1[y][0]
        y_depth = self.dict1[y][1]
        
        return x_depth == y_depth and x_parent != y_parent
```
解法二：思路同上，只不过用递归DFS遍历

* [301 Remove Invalid Parentheses](https://leetcode.com/problems/remove-invalid-parentheses/)
```python
class Solution(object):
    def isValid(self, head):
        count =0
        for shead in head:
            if shead == '(':
                count += 1
            if shead == ')':
                count -= 1
            if count < 0:
                return False
        return count == 0
    
    def removeInvalidParentheses(self, s):
        """
        :type s: str
        :rtype: List[str]
        """
        visited = set()
        q = collections.deque()
        res = []
        found = False

        q.append(s)
        visited.add(s)

        while q:
            head = q.popleft()
            if self.isValid(head):
                res.append(head)
                found = True
                visited.add(head)
            if found:
                continue
            # 入队
            for i in range(len(head)):
                if head[i] == '(' or head[i] == ')':
                    news = head[0:i] + head[i + 1:]
                    if news not in visited:
                        q.append(news)
                        visited.add(news)

        return res
```
解法一：BFS的思路，首先将s入队，然后出队，判断队头字符串是否是合法的，如果是合法的，就不继续往后追加（因为求得是最少去掉几个字符）；如果不合法，就从头开始，依次去掉队头元素的每个字符，入队，待下一次出队判断<br>
注意：用了一个set集合，记录被判断过的字符串，防止重复判断.每次入队的时候，就在set中检查一下，没有出现过的才入队

```python
class Solution(object):
    def isValid(self, s):
        count =0
        for ss in s:
            if ss == '(':
                count += 1
            if ss == ')':
                count -= 1
            if count < 0:
                return False
        return count == 0
    
    
    def dfs(self, s, start, cnleft, cnright, res):
        if cnleft == 0 and cnright == 0:
            if self.isValid(s):
                res.append(s)
        for i in range(start, len(s)):
            if i != start and s[i] == s[i-1]:
                continue
            if s[i] == '(' and cnleft > 0:
                self.dfs(s[:i]+s[i+1:], i, cnleft-1, cnright, res)
            if s[i] == ')' and cnright > 0:
                self.dfs(s[:i]+s[i+1:], i, cnleft, cnright-1, res)
                
                
    def removeInvalidParentheses(self, s):
        """
        :type s: str
        :rtype: List[str]
        """
        # 统计多余的左右括号数目cnleft,cnright
        res = []
        cnleft, cnright = 0, 0
        for i in range(len(s)):
            if s[i] == '(':
                cnleft += 1
            elif s[i] == ')' and cnleft > 0:
                cnleft -= 1
            elif s[i] == ')' and cnleft <= 0:
                cnright += 1
        self.dfs(s, 0, cnleft, cnright, res)
        return res
```
解法二：DFS思路 首先统计多余的左括号和右括号的个数（如“(()))”中多余的左括号个数是0，多余的右括号个数是1），然后从头开始遍历s，如果当前遇到的是左括号，且不是第一次遇到，就去掉当前的左括号，从当前位置继续遍历；同理右括号也是。如果多余的左右括号个数都为0，检查字符串是否合法，合法就加入结果<br>
需要注意的是：不是遇到左括号就去掉，比如‘(()’，去掉一个左括号个去掉第二个左括号效果是一样的，为了防止重复检查，遇到第一个左括号的时候，继续遍历，直到遇到最后一个左括号的时候，才去掉

* [279 Perfect Squares](https://leetcode.com/problems/perfect-squares/)
```python
import collections
class Solution(object):
    def numSquares(self, n):
        """
        :type n: int
        :rtype: int
        """

        i = 1
        sqnum = []
        while i*i <= n: # 首先列出来备选的平方数列表
            sqnum.append(i*i)
            i += 1
        
        depth = 1
        q = collections.deque()
        visited = set() # set控制只有那些没有出现过的数字才可入队

        for i in range(len(sqnum)): 
            if sqnum[i] == n: # 特例[4]等
                return 1
            else:
                q.append([sqnum[i], n - sqnum[i]])  # 首先把所有元素:剩下的值 入队
                visited.add(n - sqnum[i])

        while q:
            size = len(q)
            for i in range(size):
                number, curn = q.popleft()
                for j in range(len(sqnum)):
                    if curn == sqnum[j]:  # 找到最后一个值
                        return depth + 1
                    if sqnum[j] < curn and (curn - sqnum[j]) not in visited:
                        q.append([sqnum[j], curn - sqnum[j]]) # 重新入队
            depth += 1
```
思路一：采用BFS的思想，树的每一层都拿所有符合条件的平方数去试<br>
注意：必须要加一个set，控制防止一个数字重复出现，因为一个数字第一次出现的时候，深度是最浅的。如果没有这个控制，当n比较大的时候，会超时
```python
import math
class Solution(object):
    def numSquares(self, n):
        """
        :type n: int
        :rtype: int
        """

        # 简化数字
        while n % 4 == 0:
            n = n // 4
       
        if n % 8 == 7:
            return 4
        # n需要一个完全平方数来表示的情况下，那么这个完全平方数一定是本身，即只需要判断n是否是平方数
        if math.pow(int(math.sqrt(n)),2) == n:
            return 1
        
        # n需要两个完全平方数表示的情况下，那么第一个完全平方数就从1开始试，看剩下的数可以表示成某个数的平方的话，就返回2 
        i = 1
        while i * i < n:
            j = math.sqrt(n-i*i)  # 第一个完全平方数是i，那么剩下的是n-i*i
            if j == int(j):
                return 2
            i += 1
            
        return 3
```
思路二：套用数学公式，直接算<br>
规则：<br>
四平方定理：任何一个数都可以表示成不超过4个平方数的和，因此本题只会返回1,2,3,4,中的任意一个数<br>
简化：如果一个数含有因子4，可以将其去除，不影响结果（如2和8返回的结果相同）<br>
如果一个数除以8余7的话，则一定是由4个平方数的和组成<br>
因此接下来尝试将其拆成两个平方数之和，如果拆成功了，返回2；拆成一个平方数，返回1；其他情况返回3

* [207 Course Schedule](https://leetcode.com/problems/course-schedule/)
```python
import collections
class Solution(object):
    def canFinish(self, numCourses, prerequisites):
        """
        :type numCourses: int
        :type prerequisites: List[List[int]]
        :rtype: bool
        """
        graph = []
        list1 = [0 for _ in range(numCourses)]
        q = collections.deque()
        # 用二维数组表示有向图([0,1]表示0的先修课是1，则在图中表示成[1,0]),并记录每个节点的入度
        for ps in prerequisites:
            graph.append([ps[1],ps[0]])
            list1[ps[0]] += 1
            
        # 将节点入度数是0的元素入队
        for i in range(len(list1)):
            if list1[i] == 0:
                q.append(i)
        
        # 遍历队列，访问其相邻元素，访问到了就将相邻元素的入度数减1，如果减去1过后是0，就加入队列中
        while q:
            cournum = q.popleft()
            for nodes in graph:
                if nodes[0] == cournum:
                    list1[nodes[1]] -= 1
                    if list1[nodes[1]] == 0:
                        q.append(nodes[1])
        # 全部遍历完，如果还有节点的入度是大于0，说明图中存在环，返回False
        if max(list1) != 0 :
            return False
        return True
```
思路1：BFS 首先将节点表示成图的形式，并记录每个节点的入度；将入度为0的节点加入到队列中；逐个元素出队，访问其相邻元素，并将其入度数-1，如果相邻元素的入度数变成0，则也加入队列。整个队列元素遍历结束后，如果还有节点的度数不为0，则说明图中有环路存在。

```python
import collections
class Solution(object):
    def dfs(self, i, visited, graph):
        if visited[i] == -1:
            return False
        if visited[i] == 1:
            return True
        visited[i] = -1
        for j in graph[i]:
            if not self.dfs(j, visited, graph):
                return False
        visited[i] = 1
        return True
        
    def canFinish(self, numCourses, prerequisites):
        """
        :type numCourses: int
        :type prerequisites: List[List[int]]
        :rtype: bool
        """
        graph = [[] for _ in range(numCourses)]
        visited = [0 for _ in range(numCourses)]
        for ps in prerequisites: # graph[i]记录i号课程的前置课程
            graph[ps[0]].append(ps[1])

        for i in range(numCourses):
            if not self.dfs(i, visited, graph):
                return False
        return True
```
思路2：DFS思想：首先用graph记录每个课程对应的前置课程；visited记录每个课程的状态：0未遍历到；1已经遍历过；-1正在遍历中。<br>
然后遍历每门课程i，如果i已经遍历过，返回true；如果i正在被遍历，说明存在环路，返回false，否则i没有被遍历到，就暂时将visited[i]的值修改为-1，表示当前正在遍历；然后对i的每门先行课进行同样的操作，若有返回false的，说明在i和i的先行课同时被遍历到了，返回false


* [102 Binary Tree Level Order Traversal](https://leetcode.com/problems/binary-tree-level-order-traversal/)
```python
import collections
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def levelOrder(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        if not root:
            return []
        res = []
        
        q = collections.deque()
        q.append(root)
        while q:
            size = len(q)
            temp = []
            for i in range(size):
                treenode = q.popleft()
                temp.append(treenode.val)
                if treenode.left:
                    q.append(treenode.left)
                if treenode.right:
                    q.append(treenode.right)
            res.append(temp)
        return res
```
思路：比较简单，就是BFS层次遍历

* [101 Symmetric Tree](https://leetcode.com/problems/symmetric-tree/)
```python
import collections
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):            
    def isSymmetric(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        q = collections.deque()
        q.append([root, root])
        
        while q:
            node1, node2 = q.popleft()
            if not node1 and not node2:
                continue
            if not node1 or not node2:
                return False
            if node1.val != node2.val:
                return False
            
            q.append([node1.left, node2.right])
            q.append([node1.right, node2.left])
        return True

```
思路：找到每一层位置对称元素node1和node2，比较两者

* [200 Number of Islands](https://leetcode.com/problems/number-of-islands/)
```python
import collections
class Solution(object):
    def numIslands(self, grid):
        """
        :type grid: List[List[str]]
        :rtype: int
        """
        if not grid:
            return 0
        
        m = len(grid)
        n = len(grid[0])
        q = collections.deque()
        # 首先把所有为1的节点加入lands
        lands = set()
        count = 0
        for i in range(m):
            for j in range(n):
                if grid[i][j] ==  '1':
                    lands.add((i, j))
        while lands:
            count += 1
            q.append(lands.pop())
            while q:
                curi,curj = q.popleft()
                # 上
                if (curi-1, curj) in lands:
                    q.append((curi-1, curj))
                    lands.remove((curi-1, curj))
                # 下
                if (curi+1, curj) in lands:
                    q.append((curi+1, curj))
                    lands.remove((curi+1, curj))
                # 左
                if (curi, curj-1) in lands:
                    q.append((curi, curj-1))
                    lands.remove((curi, curj-1))
                # 右
                if (curi, curj+1) in lands:
                    q.append((curi, curj+1))
                    lands.remove((curi, curj+1))
        return count
```
思路：首先将所有值为1的坐标对加入到一个集合里面；然后逐个将集合里的元素弹出，入队，并且判断当前位置四个方向的坐标对是否在集合里，在的话就将其从集合中去除，并入队。直到对为空的时候，说明产生了一个区域。count+1，然后继续将集合元素入队，开始下一个区域的判断。

```python
class Solution(object):
    def dfs(self, grid, i, j):
        if i < 0 or j < 0 or i > len(grid)-1 or j > len(grid[0])-1 or grid[i][j] != '1':
        # 判断越界：if i == -1 or j == -1 or i == len(grid) or j == len(grid[0]) or grid[i][j] != '1':
            return 
        grid[i][j] = '#' # 标记已经访问过
        self.dfs(grid, i, j - 1)
        self.dfs(grid, i, j + 1)
        self.dfs(grid, i - 1, j)
        self.dfs(grid, i + 1, j)
        
    def numIslands(self, grid):
        """
        :type grid: List[List[str]]
        :rtype: int
        """
        if not grid:
            return 0
        
        m = len(grid)
        n = len(grid[0])
        count = 0 
        for i in range(m):
            for j in range(n):
                if grid[i][j] == '1':
                    self.dfs(grid, i, j)
                    count += 1
        return count
```
思路二：采用DFS的思想，逐个遍历；对于每个grid[i][j]，首先标记其访问过了，然后递归判断其四个方向，递归终止条件为下标越界，或者取值不为1，表示此处有隔断，直接返回。每返回一次，表示找到一块区域，count计数加1
