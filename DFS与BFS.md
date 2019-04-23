## DFS
* 图示：
![image](https://github.com/MissAquarius/ForJobHunting/blob/master/image/DFS.png)


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
解法一：BFS的思路，明天补充吧~

