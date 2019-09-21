## DFS
* 图示：
![image](https://github.com/MissAquarius/ForJobHunting/blob/master/image/DFS.png)

* 例题：
  * [301 remove-invalid-parentheses](https://leetcode.com/problems/remove-invalid-parentheses/)（见BFS部分其他解法）
  * [207 Course Schedule](https://leetcode.com/problems/course-schedule/) （见BFS部分其他解法）
  * [200 Number of Islands](https://leetcode.com/problems/number-of-islands/)（见BFS部分其他解法）
  * [101 Symmetric Tree](https://leetcode.com/problems/symmetric-tree/)（见BFS部分其他解法）
  
  * [494 Target Sum](https://leetcode.com/problems/target-sum/)
```python
class Solution(object):
    def dfs(self, CurIndex, CurTotal, S, nums): 
        if CurIndex == len(nums):
            if CurTotal == S:
                self.count += 1
            return 
        
        self.dfs(CurIndex + 1, CurTotal + nums[CurIndex], S, nums)
        self.dfs(CurIndex + 1, CurTotal - nums[CurIndex], S, nums)
        
    def findTargetSumWays(self, nums, S):
        """
        :type nums: List[int]
        :type S: int
        :rtype: int
        """
        if not nums:
            return 0
        
        self.count = 0
        self.dfs(0, 0, S, nums)
        return self.count
```
思路一：DFS，相当于遍历所有可能的结果，超时……所以这种“只求有几种方法，而不求具体每种方法是什么”的问题，用DFS会超时，建议用DP
```python
待补充
```
思路二：dp

* [394 Decode String](https://leetcode.com/problems/decode-string/)
```python
class Solution(object):
    def decodeString(self, s):
        """
        :type s: str
        :rtype: str
        """
        stack = []
        for i in range(len(s)):
            if s[i] != ']':
                stack.append(s[i])
            else:
                temp = []
                while stack and stack[-1] != '[':
                    temp.append(stack.pop())
                
                temps = ''
                for i in range(len(temp)-1,-1,-1):
                    temps += temp[i]

                stack.pop() # 弹出[
                # 寻找次数
                times = ''
                while stack and stack[-1] in '0123456789' :
                    times += stack.pop()
                times = int(times[-1::-1])
                stack.append(temps * times)
                
        res = ''
        for s in stack:
            res += s
        return res
```
思路：借助于栈，遇到非']'就进栈，遇到']'，就开始退栈，直到遇到'[', 取出来的元素就是要重复的元素；重复的次数就继续退栈，直到遇到非数字；然后转成int型；最后将重复后的元素压栈，继续遍历

```python
class Solution(object):
    def decodeString(self, s):
        """
        :type s: str
        :rtype: str
        """
        stack = []
        res = ''
        for i in range(len(s)):
            if s[i] == '[':
                stack.append(i)
            elif s[i] == ']':
                startindex = stack.pop()
                if not stack:
                    temp = self.decodeString(s[startindex+1:i])  # 要重复的元素
                    # 寻找重复次数
                    numstart = startindex
                    while numstart - 1 >= 0 and s[numstart-1].isdigit():
                        numstart -= 1
                    times = int(s[numstart:startindex])
                    res += temp * times
            elif not stack and s[i].isalpha():
                res += s[i]
        return res
```
思路：难点在于递归的想法，递归用于寻找最外层的[] 对的位置（为什么？这一点第二遍做题的时候卡了好久，因为注意看递归上层的if条件是stack为空的时候，才进行递归，那么如果stack为空，说明最先入栈的左括号位置被弹出），找到后就递归寻找[]框起来的元素，temp是其返回的结果；stack中入的是下标，而不是具体的元素值

* [213. House Robber](https://leetcode.com/problems/house-robber/)
```python

```
动态规划，待补充
* [213. House Robber II](https://leetcode.com/problems/house-robber-ii/)
```python

```
动态规划，待补充
* [337 House Robber III](https://leetcode.com/problems/house-robber-iii/)
```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def dfs(self, curnode):
        curarr = [0, 0]
        if curnode == None:
            return curarr
        # curarr[0]表示抢劫当前节点，curarr[1]表示不抢劫当前节点
        leftarr = self.dfs(curnode.left)
        rightarr =  self.dfs(curnode.right)
        curarr[0] = curnode.val + leftarr[1] + rightarr[1]
        curarr[1] = max(leftarr[0], leftarr[1]) + max(rightarr[0], rightarr[1])
        return curarr
    def rob(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        arr = self.dfs(root)
        return max(arr)
```
思路：为每个node寻找其抢劫和不抢劫情况下的最大值, 递归过程如下<br>
抢劫的情况：其获得值 = 其值 + 左孩子不抢劫的最大值 + 右孩子不抢劫的最大值<br>
不抢劫的情况： 其获得值 = max(左子树抢劫, 左子树不抢劫) + max(右孩子抢劫, 右孩子不抢劫) <br>
终止条件：如果遇到节点为None的情况，说明抢劫和不抢劫都返回0  <br>
每次遇到递归就无从下手，其实递归不要考虑具体的实现细节，只写出在当前情况的操作流程即可，比如如何求左孩子、右孩子在两种情况下的最值不需要考虑，因为递归操作，只需要拿到其结果即可。把递归当成是一个处理流程。

* [124 Binary Tree Maximum Path Sum](https://leetcode.com/problems/binary-tree-maximum-path-sum/)
```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def dfs(self, node):
        if not node:
            return 0

        leftmax = max(self.dfs(node.left), 0) # 左子树返回
        rightmax = max(self.dfs(node.right), 0) # 右子树返回
        local_max = leftmax + rightmax + node.val  # 局部最大值取：当前数的所有值
        self.global_max = max(self.global_max, local_max )  # 更新全局最大值
        return max(leftmax, rightmax) + node.val  # 递归函数返回的是左边-节点或者右边-节点的最大值
        
    def maxPathSum(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        
        self.global_max = float('-inf')
        self.dfs(root)
        return self.global_max
         
```
思路：全局最大值：不一定非要取到root节点，有可能中间的某个子树就取得最大值；局部最大值是当前递归到的局部树：用其左子树返回值、右子树返回值和当前根值相加<br>
比较绕的是：在回溯过程中，只能从左子树回溯到根节点，或者右子树回溯到根节点，所以递归的返回值取左右子树的最大值与当前根节点之和<br>
即：递归函数返回的是当前子树的最大值，其要么是从左孩子到根，要么是右孩子到根

* [104. Maximum Depth of Binary Tree](https://leetcode.com/problems/maximum-depth-of-binary-tree/)
```python
import collections
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def maxDepth(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        if not root:
            return 0
        
        q = collections.deque()
        depth = 0
        q.append(root)
        
        while q:
            for i in range(len(q)):
                node = q.popleft()
                if node.left:
                    q.append(node.left)
                if node.right:
                    q.append(node.right)
            depth += 1
            
        return depth
```
求二叉树的高度：BFS思想；将每一层入队遍历，每遍历一轮，层数就加1，且将左右孩子入队

```python
import collections
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def dfs(self, node):
        if not node:
            return 0
        left_height = self.dfs(node.left)
        right_height = self.dfs(node.right)
        return max(left_height, right_height) + 1
    
    def maxDepth(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        return self.dfs(root)
```
思路二：DFS思想，对于每个节点，求其左右子树高度的最大值，然后加1便是以当前节点为根的树的高度

* [114 Flatten Binary Tree to Linked List](https://leetcode.com/problems/flatten-binary-tree-to-linked-list/)
```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def dfs(self, node):
        if node:
            self.reslist.append(node)
            self.dfs(node.left)
            self.dfs(node.right)
            
    def flatten(self, root):
        """
        :type root: TreeNode
        :rtype: None Do not return anything, modify root in-place instead.
        """
        # 二叉树的前序遍历？？
        # 递归版
        self.reslist = []
        if not root:
            return []
        self.dfs(root)
        for i in range(len(self.reslist)-1):
            self.reslist[i].left = None
            self.reslist[i].right = self.reslist[i+1]
        root = self.reslist[0]
```
思想：这道题基于二叉树的前序遍历，只不过将遍历后的节点都存储到节点的右孩子。用DFS思想递归遍历，不好的地方在于用了额外的存储空间暂存遍历顺序，最后重排
```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def flatten(self, root):
        """
        :type root: TreeNode
        :rtype: None Do not return anything, modify root in-place instead.
        """
        # 前序遍历 非递归
        if not root:
            return []
        stack = []
        reslist = []
        p = root
        while stack or p:
            if p:
                reslist.append(p)
                stack.append(p)
                p = p.left
            elif stack:
                p = stack.pop()
                p = p.right

        for i in range(len(reslist)-1):
            reslist[i].left = None
            reslist[i].right = reslist[i+1]
        root = reslist[0]
 ```
 思想：二叉树前序遍历的非递归算法

* [105. Construct Binary Tree from Preorder and Inorder Traversal](https://leetcode.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/)
```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def helper(self, preorder, inorder):
        if inorder:
            value = preorder.pop(0)  # 当前根节点值
            root = TreeNode(value)
            temp_index = inorder.index(value) # 找到其值在中序遍历中的位置，将遍历结果分为两半
            root.left = self.helper(preorder, inorder[:temp_index])
            root.right = self.helper(preorder, inorder[temp_index+1:])
            return root
    def buildTree(self, preorder, inorder):
        """
        :type preorder: List[int]
        :type inorder: List[int]
        :rtype: TreeNode
        """
        # 给定前序和中序遍历的结果，构造出整棵树
        return self.helper(preorder, inorder) 
```
思路：前序遍历确定根节点的值，拿到根节点值后可将中序遍历的结果分为两半，位于值左边的说明是左子树上的元素，位于右边的是右子树的元素；对于左右两边，再根据前序遍历与中序遍历的结果找到树，递归求解

* [98. Validate Binary Search Tree](https://leetcode.com/problems/validate-binary-search-tree/)
```python
import collections
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def PreOrderRecursive(self, node):
        if node:
            self.PreOrderRecursive(node.left)
            self.res.append(node.val)
            self.PreOrderRecursive(node.right)
            
    def isValidBST(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        # 二叉搜索树，根据其性质对其进行中序遍历，会得到一个升序的序列
        # 递归算法
        self.res = []
        self.PreOrderRecursive(root)
        
        for i in range(len(self.res)-1):
            if self.res[i] >= self.res[i+1]:
                return False
        return True
```
```python
class Solution(object):
    def isValidBST(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        # 中序遍历，非递归
        if not root:
            return True
        stack = []
        res = []
        p = root
    
        while stack or p:
            if p:   # 指针非空，进站，向左走
                stack.append(p)
                p = p.left
            else:   # 指针为空，退栈，访问，向右走
                p = stack.pop()
                res.append(p.val)
                p = p.right
        
        for i in range(len(res)-1):
            if res[i] >= res[i+1]:
                return False
        return True
```
思路：判断一颗树是否是二叉搜索树<br>
二叉搜索树的定义：对于每一个节点，其左孩子（若有）的值小于节点值，右孩子（若有）的值大于节点值；同时每一个子节点都是一颗BST<br>
注意：二叉搜索树中，对于一个节点来说，不仅是左孩子节点的值要小于节点值，而且孙子节点等值都要小于其值（这一点之前竟然忘了，导致第一次没有ac），也就是说位于节点左边的值都要小于其值；右孩子同理。<br>
性质：对一颗BST树进行中序遍历，会得到一个递增的有序序列；位于树最左端的值是最小的，位于树最右边的值是最大的。<br>
因此本题一种思路是中序遍历树，看是否可以得到一个递增序列
```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def isValid(self, node, curmin, curmax):
        if not node:
            return True
        if node.val <= curmin or node.val >= curmax:
            return False
        return self.isValid(node.left,curmin, node.val) and self.isValid(node.right, node.val, curmax)
    
    def isValidBST(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        return self.isValid(root, float("-inf"), float("inf"))
 ```
 思路：对于每一个节点，都有一个范围[min, max]，如节点node，其左孩子的最大值不能超过node值，右节点的最小值不能小于node值，递归判断。初始时，取整数中的最值，子啊python里用float(-inf)代替C++中的MIN_INT，float(inf)代替MAX_INT
 
[112. Path Sum](https://leetcode.com/problems/path-sum/)
```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def dfs(self, node, sum):
        if not node:
            return False
        if node.val == sum and not node.left and not node.right: # 说明到叶子节点了
            return True  
        sum -= node.val 
        return self.dfs(node.left, sum) or self.dfs(node.right, sum)
    
    def hasPathSum(self, root, sum):
        """
        :type root: TreeNode
        :type sum: int
        :rtype: bool
        """
        return self.dfs(root, sum)
```
思路：对于每一个节点，递归遍历其左右节点，只要有一个返回True即可；每次递归时，sum值就要减去当前节点值，作为新的目标sum，直到遇到叶子节点的时候，看当前sum是否与叶子节点值相同

```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):    
    def hasPathSum(self, root, sum):
        """
        :type root: TreeNode
        :type sum: int
        :rtype: bool
        """
        if not root:
            return False
        stack = []
        stack.append((root, root.val))

        while stack:
            curnode,val = stack.pop()
            if not curnode.left and not curnode.right:
                if val == sum:
                    return True
            # 此处先加右节点因为栈是先进后出，这样不断出栈，可一直遍历左节点
            if curnode.right:
                stack.append((curnode.right, curnode.right.val+val))
            if curnode.left:
                stack.append((curnode.left, curnode.left.val+val))
            
        return False
```
思路：DFS的非递归遍历，栈中记录每个节点及当前新的sum值（=原sum值加上当前节点的val），入栈顺序是先右再左，是为了出栈的时候可以先左再右，达到DFS的要求。

* [113. Path Sum II](https://leetcode.com/problems/path-sum-ii/)
```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def pathSum(self, root, sum):
        """
        :type root: TreeNode
        :type sum: int
        :rtype: List[List[int]]
        """
        if not root:
            return []
        
        res = []
        stack = []
        stack.append((root, root.val, [root.val]))
        
        while stack:
            node, val, ls = stack.pop()
            if not node.left and not node.right:
                if val == sum:
                    res.append(ls)
            if node.right:
                stack.append((node.right, node.right.val + val, ls + [node.right.val]))
            if node.left:
                stack.append((node.left, node.left.val + val ,ls + [node.left.val]))
        return res
 ```
 思路：同上一样，只不过栈中额外保存当前加入新sum值的node值，注意此处不可直接改变ls，因为在right改变了，那在left会再次改变。要保证二者是独立的，就用+的写法。

* [108. Convert Sorted Array to Binary Search Tree](https://leetcode.com/problems/convert-sorted-array-to-binary-search-tree/)
```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def helper(self, nums): 
        if not nums:
            return None
        mid = len(nums) // 2
        root = TreeNode(nums[mid])
        root.left = self.helper(nums[:mid])
        root.right = self.helper(nums[mid+1:])
        return root
    def sortedArrayToBST(self, nums):
        """
        :type nums: List[int]
        :rtype: TreeNode
        """
        if not nums:
            return None
        return self.helper(nums)
  ```
  思路：去当前nums的中点作为根节点的值；左边的节点位于左子树上，右边的位于右子树上，递归左右子树即可
  
* [329. Longest Increasing Path in a Matrix](https://leetcode.com/problems/longest-increasing-path-in-a-matrix/)
```python
class Solution(object):
    def dfs(self, matrix, dp, i, j, m, n):
        if dp[i][j]:
            return dp[i][j]
        
        maxlen = 1 # 初始时，matrix[i][j]算在内，所以初值为1，该值记录matrix[i][j]为起始节点的最长递增序列长度
        left_lenth, right_lenth, up_lenth, down_lenth = 0, 0, 0, 0
        # 递归求其四个方向上的最长递增序列长度，并且及时更新最大值
        if j-1 >= 0 and matrix[i][j-1] > matrix[i][j]:
            left_lenth = 1 + self.dfs(matrix, dp, i, j-1, m, n)
            # maxlen = max(maxlen, left_lenth)
        if j+1 < n and matrix[i][j+1] > matrix[i][j]:
            right_lenth = 1 + self.dfs(matrix, dp, i, j+1, m, n)
            # maxlen = max(maxlen, right_lenth)
        if i-1 >= 0 and matrix[i-1][j] > matrix[i][j]:
            up_lenth = 1 + self.dfs(matrix, dp, i-1, j, m, n)
            # maxlen = max(maxlen, up_lenth)
        if i+1 < m and matrix[i+1][j] > matrix[i][j]:
            down_lenth = 1 + self.dfs(matrix, dp, i+1, j, m, n)
            # maxlen = max(maxlen, down_lenth)
          
        maxlen = max(maxlen, left_lenth, right_lenth, up_lenth, down_lenth )       
        # 更新维护的dp数组
        dp[i][j] = maxlen
        return maxlen
        
    def longestIncreasingPath(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: int
        """
        if not matrix :
            return 0
        m = len(matrix)
        n = len(matrix[0])
        dp = [[0 for i in range(n)] for j in range(m)] # 存储以matrix[i][j]为起点的最大递增长度
        maxlen = 0
        
        
        
        # 对于每个节点，都要计算以该节点起始的最长递增序列长度
        for i in range(m):
            for j in range(n):
                maxlen = max(maxlen, self.dfs(matrix, dp, i, j, m, n))
        return maxlen
 ```
 思路：dp+dfs
 疑惑点：对于每个方向的递归，为什么求得该方向上的最长递增序列的值的时候，要立即与maxlen比较；而不能等四个方向都递归完了，集中比较（如注释的那句代码，即在四个方向返回的值和maxlen中取最值），报错信息为：left_lenth在使用之前未定义？感觉是对递归中局部变量的存储不理解？？之前是觉得互不干涉的，只需关注当前操作即可，不必关心在递归过程中对这个值的操作
 
```python
```
纯dp方法，待补充

* [130. Surrounded Regions](https://leetcode.com/problems/surrounded-regions/)
```python
import collections
class Solution(object):  
    def dfs(self, board, i, j, m ,n):
        board[i][j] = '*'
        if  i -1 >= 0 and board[i-1][j] == 'O':
            self.dfs(board, i-1, j, m, n)
        if  i + 1 < m and board[i+1][j] == 'O':
            self.dfs(board, i+1, j, m, n)        
        if  j -1 >= 0 and board[i][j-1] == 'O':
            self.dfs(board, i, j-1, m, n)     
        if  j +1 < n and board[i][j+1] == 'O':
            self.dfs(board, i, j+1, m, n)       
    
    def solve(self, board):
        """
        :type board: List[List[str]]
        :rtype: None Do not return anything, modify board in-place instead.
        """
        # BFS
        if not board:
            return []
        
        m = len(board)
        n = len(board[0])
        q = collections.deque()
        for i in range(m):
            for j in range(n):
                if  board[i][j] == 'O' and (i == 0 or i == m-1 or j == 0 or j == n-1):
                    self.dfs(board, i, j, m, n)
  
        for i in range(m):
            for j in range(n):
                if board[i][j] == 'O':
                    board[i][j] = 'X'
                if board[i][j] == '*':
                    board[i][j] = 'O'
```
思想一：DFS，对于每个边界上的值，如果值为'O',就调用DFS算法，首先将该位置元素值改为某个特殊标记，然后递归判断其上下左右四个方位有没有元素还是'O'<br>
最终将剩下的'O'换成‘X’，将特殊标记换回来为'O'

```python
import collections
class Solution(object):  
    def bfs(self, board, q, m ,n): 
        while q:
            i, j = q.popleft()
            board[i][j] = '*'
            if  i -1 >= 0 and board[i-1][j] == 'O':
                    q.append((i-1, j))
            if  i + 1 < m and board[i+1][j] == 'O':
                    q.append((i+1, j))
            if  j -1 >= 0 and board[i][j-1] == 'O':
                    q.append((i, j-1))
            if  j +1 < n and board[i][j+1] == 'O':
                    q.append((i, j+1))
    
    def solve(self, board):
        """
        :type board: List[List[str]]
        :rtype: None Do not return anything, modify board in-place instead.
        """
        # BFS
        if not board:
            return []
        
        m = len(board)
        n = len(board[0])
        q = collections.deque()
        for i in range(m):
            for j in range(n):
                if  board[i][j] == 'O' and (i == 0 or i == m-1 or j == 0 or j == n-1):
                    #说明在边界上有O
                    q.append((i, j))
        self.bfs(board, q, m, n)
    
        for i in range(m):
            for j in range(n):
                if board[i][j] == 'O':
                    board[i][j] = 'X'
                if board[i][j] == '*':
                    board[i][j] = 'O'
```    
思想二：BFS，首先将所有边界上取值为‘O’的位置入队；然后开始逐个出队，并且暂且将该位置的元素改为其他标记，之后判断这些位置的上下左右四个方位是否也是'O',是的话就继续入队，直到队列为空。<br>
最终将剩下的'O'换成‘X’，将特殊标记换回来为'O'

* [116. Populating Next Right Pointers in Each Node](https://leetcode.com/problems/populating-next-right-pointers-in-each-node/)
```python
import collections
"""
# Definition for a Node.
class Node(object):
    def __init__(self, val, left, right, next):
        self.val = val
        self.left = left
        self.right = right
        self.next = next
"""
class Solution(object):
    def connect(self, root):
        """
        :type root: Node
        :rtype: Node
        """
        q = collections.deque()
        if not root:
            return None
        q.append(root)
        while q:
            length = len(q)
            for i in range(length):
                node = q.popleft()
                if i == length-1:
                    node.next = None
                else:
                    node.next = q[0]
                if node.left:
                    q.append(node.left)
                if node.right:
                    q.append(node.right)
        return root
```
思路：BFS，感觉没啥意义的一道题，就是把一颗二叉树中同一层次的节点之间用next指针相连

```python
"""
# Definition for a Node.
class Node(object):
    def __init__(self, val, left, right, next):
        self.val = val
        self.left = left
        self.right = right
        self.next = next
"""
class Solution(object):
    def connect(self, root):
        """
        :type root: Node
        :rtype: Node
        """
        if not root:
            return None
        
        stack = []
        stack.append(root)
        while stack:
            node = stack.pop()
            if node.left and node.right:
                node.left.next = node.right
                if node.next:
                    node.right.next = node.next.left
                stack.append(node.right)
                stack.append(node.left)
        return root
 ```
 思路：DFS思想
 
* [210. Schedule II](https://leetcode.com/problems/course-schedule-ii/)
```python
import collections
class Solution(object):
    def findOrder(self, numCourses, prerequisites):
        """
        :type numCourses: int
        :type prerequisites: List[List[int]]
        :rtype: List[int]
        """
        graph = []
        list1 = [0 for _ in range(numCourses)] # 记录对应n号课程的入度
        q = collections.deque()
        res = []
        
        for ps in prerequisites:
            graph.append((ps[1], ps[0]))
            list1[ps[0]] += 1
        
        
        for i in range(len(list1)):
            if list1[i] == 0:
                q.append(i)
        
        while q:
            cournum = q.popleft()
            res.append(cournum)
            for node in graph:
                if node[0] == cournum:  
                    list1[node[1]] -= 1
                    if list1[node[1]] == 0:
                        q.append(node[1])
                        
        if max(list1) != 0:
            return []
       
        return res
```
思路：与I的差别是 在出队的时候，保存当前出队的元素编号，这就是最终访问的顺序

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

* [102. Binary Tree Level Order Traversal] (https://leetcode.com/problems/binary-tree-level-order-traversal/)
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
思路：BFS思想，对于每一层，找到每一层位置对称元素node1和node2，比较两者，比较结束后，再把对称位置（下一次需要比较的元素对）入队。


```python
import collections
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):  
    def dfs(self, node1, node2):
        if not node1 and not node2:
            return True
        if not node1 or not node2:
            return False
        if node1.val != node2.val:
            return False
        return self.dfs(node1.left, node2.right) and self.dfs(node1.right, node2.left)
    
    def isSymmetric(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        if not root:
            return True
        return self.dfs(root.left, root.right)

```
思想：DFS思想，用了递归的方法

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
