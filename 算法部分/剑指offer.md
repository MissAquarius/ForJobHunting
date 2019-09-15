## 链表（八道）

* 剑指Offer（三）：从尾到头打印链表
```python
# -*- coding:utf-8 -*-
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    # 返回从尾部到头部的列表值序列，例如[1,2,3]
    def printListFromTailToHead(self, listNode):
        # write code here
        stack = []
        while listNode:
            stack.append(listNode.val)
            listNode = listNode.next
        res = []
        while stack:
            res.append(stack.pop())
        return res
```
借助栈先进后出的特性，将链表元素逐个入栈，最后再出栈

* 剑指Offer（十四）：链表中倒数第k个结点
```python
class Solution:
    def FindKthToTail(self, head, k):
        # write code here
        if not head or k <= 0:
            return None
        dummy = ListNode(-1)
        dummy.next = head
        fast = slow = dummy
        for _ in range(k):
            fast = fast.next
            if not fast:
                return None
        while fast.next:
            fast = fast.next
            slow = slow.next
        return slow.next
```
同Leetcode中删除倒数第K个节点<br>
注意：k值可能比链表的长度还要大，因此fast指针那里有判断；如果fast指针指向最后一个节点，那么slow指针的下一个节点即是倒数第k个节点

* 剑指Offer（十五）：反转链表

```python
# -*- coding:utf-8 -*-
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    # 返回ListNode
    def ReverseList(self, pHead):
        # write code here
        # 递归版
        if not pHead or not pHead.next:
            return pHead
        node = self.ReverseList(pHead.next)
        pHead.next.next = pHead
        pHead.next = None
        return node 
```
```python
# -*- coding:utf-8 -*-
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
class Solution:
    # 返回ListNode
    def ReverseList(self, pHead):
        # write code here
        # 迭代
        dummy = ListNode(-1)
        dummy.next = pHead
        if not pHead:
            return pHead
        cur = pHead.next
        pHead.next = None
        while cur:
            temp = cur.next
            cur.next = dummy.next
            dummy.next = cur
            cur = temp
        return dummy.next
```

* 剑指Offer（十六）：合并两个排序的链表
```python
# -*- coding:utf-8 -*-
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
class Solution:
    # 返回合并后列表
    def Merge(self, pHead1, pHead2):
        # write code here
        dummy = ListNode(-1)
        dummy.next = None
        temp = dummy
        while pHead1 and pHead2:
            if pHead1.val <= pHead2.val:
                temp.next = pHead1
                pHead1 = pHead1.next
            else:
                temp.next = pHead2
                pHead2 = pHead2.next
            temp = temp.next
        if pHead1:
            temp.next = pHead1
        else:
            temp.next = pHead2
        return dummy.next
```
* 剑指Offer（二十五）：复杂链表的复制
```python

```

* 剑指Offer（三十六）：两个链表的第一个公共结点
```python
# -*- coding:utf-8 -*-
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
class Solution:
    def FindFirstCommonNode(self, pHead1, pHead2):
        # write code here
        s = set()
        while pHead1:
            s.add(pHead1)
            pHead1 = pHead1.next
        while pHead2:
            if pHead2 in s:
                return pHead2
            else:
                pHead2 = pHead2.next
        return None
```
我的思路：遍历第一个链表，将节点加入set中；遍历第二个链表，同时判断节点是否在set中，不在就继续遍历，在的话就是第一个公共节点

```python
# -*- coding:utf-8 -*-
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
class Solution:
    def FindFirstCommonNode(self, pHead1, pHead2):
        # write code here
        p1 = pHead1
        p2 = pHead2
        while p1 != p2:
            if p1:
                p1 = p1.next
            else:
                p1 = pHead2
            if p2:
                p2 = p2.next
            else:
                p2 = pHead1
        return p1
```
其他解法：由于第一个公共节点之后的节点都相同，可以把两个链表拼起来：第一个链表+第二个链表，第二个链表+第一个链表，这样两个链表长度相同。用两个指针从头开始遍历，就可以找到公共节点。时间复杂度O(m+n),空间复杂度O(1)

```python
# -*- coding:utf-8 -*-
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
class Solution:
    def GetLinkedListLength(self, head):
        length = 0
        while head:
            head = head.next
            length += 1
        return length
    def FindFirstCommonNode(self, pHead1, pHead2):
        # write code here
        len1 = self.GetLinkedListLength(pHead1)
        len2 = self.GetLinkedListLength(pHead2)
        diff = abs(len1 - len2)
        if len1 > len2:
            p = pHead1
            q = pHead2
        else:
            p = pHead2
            q = pHead1
        for _ in range(diff):
            p = p.next
        while p != q and p and q:
            p = p.next
            q = q.next
        return p
```
先让长的链表把头部多出来的部分"砍掉"，然后两个指针再一起遍历，跟上面的思路大概一样。

* 剑指Offer（五十五）：链表中环的入口结点
```python
# -*- coding:utf-8 -*-
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
class Solution:
    def EntryNodeOfLoop(self, pHead):
        # write code here
        slow = fast = pHead
        flag = False
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            if slow == fast:
                flag = True
                break
        if flag:
            fast = pHead
            while fast != slow:
                fast = fast.next
                slow = slow.next
            return fast
        else:
            return None
```
同leetcode原题，此处有公式，可以画图手推一下

* 剑指Offer（五十六）：删除链表中重复的结点
```python
# -*- coding:utf-8 -*-
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
class Solution:
    def deleteDuplication(self, pHead):
        # write code here
        dummy = ListNode(-1)
        dummy.next = pHead
        pre = dummy
        cur = pHead
        
        while cur:
            if cur.next and cur.val == cur.next.val:
                while cur.next and cur.next.val == cur.val:
                    cur = cur.next
                pre.next = cur.next
                cur = pre.next
            else:
                pre = cur
                cur = cur.next
        return dummy.next
```
注意看题：排序数组，所以下一个节点值一定是大于等于当前节点值，只需要判断相等的时候，一直往后遍历，直到不相等的出现，修改指针指向即可。

## 二叉树
* 剑指Offer（四）：重建二叉树
```python
# -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    # 返回构造的TreeNode根节点    
    def reConstructBinaryTree(self, pre, tin):
        # write code here
        if len(tin) == 0:
            return None
        root = TreeNode(pre[0])
        index = tin.index(root.val)
        root.left = self.reConstructBinaryTree(pre[1:index+1], tin[:index])
        root.right = self.reConstructBinaryTree(pre[index+1:], tin[index+1:])
        return root
        
```
根据前序遍历和中序遍历的结果，重建二叉树，

* 剑指Offer（十七）：树的子结构
```python
class Solution:
    def helper(self, node1, node2):
        if not node2: # 说明node2比较完了
            return True
        if not node1: # 说明node1比较完了,node2还没完
            return False
        if node1.val != node2.val:
            return False
        return self.helper(node1.left, node2.left) and self.helper(node1.right, node2.right)

    def HasSubtree(self, pRoot1, pRoot2):
        flag = False
        if pRoot1 and pRoot2:
            if pRoot1.val == pRoot2.val:  # 找到一个节点相同，就去判断以此节点为根节点的的结构
                flag = self.helper(pRoot1, pRoot2)
            if not flag: # 如果未找到，就判断左孩子
                flag = self.HasSubtree(pRoot1.left, pRoot2)
            if not flag: # 同上
                flag = self.HasSubtree(pRoot1.right, pRoot2)
        return flag
```
思路：遍历二叉树中的每个节点，如果与待比较的节点相同，就用辅助函数判断其左右孩子是否相同，递归判断。注意的是：helper函数中的前两个if顺序不能换，要是换的话，得加判断条件，宗旨就是：如果node2比较完了，就返回True；如果node2没比较完，node1却比较完了，就返回False

* 剑指Offer（十八）：二叉树的镜像
```python
class Solution:
    # 返回镜像树的根节点
    def Mirror(self, root):
        # write code here
        if not root:
            return root
        root.left, root.right = root.right, root.left
        self.Mirror(root.left)
        self.Mirror(root.right)
        return root
```
思路：交换的时候，子节点下面的节点也交换了，画个图就知道每次其实只用交换节点的left和right即可. [Leetcode101](https://github.com/MissAquarius/ForJobHunting/blob/master/%E7%AE%97%E6%B3%95%E9%83%A8%E5%88%86/DFS%E4%B8%8EBFS.md)是类似题目，那个是判断是否是对称树。

* 剑指Offer（二十二）：从上往下打印二叉树
```python
# -*- coding:utf-8 -*-
import collections
class Solution:
    # 返回从上到下每个节点值列表，例：[1,2,3]
    def PrintFromTopToBottom(self, root):
        # write code here
        if not root:
            return []
        q = collections.deque()
        q.append(root)
        res = []
        while q:
            node = q.popleft()
            res.append(node.val)
            if node.left:
                q.append(node.left)
            if node.right:
                q.append(node.right)
        return res
```
思路：二叉树的BFS算法

* 剑指Offer（二十四）：二叉树中和为某一值的路径
```python
class Solution:
    # 返回二维列表，内部每个列表表示找到的路径
    def dfs(self, node, target, arr):
        if not node.left and not node.right and node.val == target:
            self.res.append(arr + [node.val])
        if node.left:
            self.dfs(node.left, target - node.val, arr+[node.val])
        if node.right:
            self.dfs(node.right, target - node.val, arr+[node.val])
            
    def FindPath(self, root, expectNumber):
        # write code here
        self.res = []
        if not root:
            return self.res
        self.dfs(root, expectNumber,[])
        self.res.sort(key = lambda x : len(x), reverse=True)
        return self.res
```

* 剑指Offer（三十八）：二叉树的深度
```python
class Solution:
    def TreeDepth(self, pRoot):
        # write code here
        if not pRoot:
            return 0
        left_height = self.TreeDepth(pRoot.left)
        right_height = self.TreeDepth(pRoot.right)
        height = max(left_height, right_height) + 1
        return height
```
思路：leetcode原题，递归即可

* 剑指Offer（三十九）：平衡二叉树
```python
class Solution:
    def FindHeight(self, node):
        if not node:
            return 0
        left = self.FindHeight(node.left)
        right = self.FindHeight(node.right)
        return  max(left, right) + 1
    
    def IsBalanced_Solution(self, pRoot):
        # write code here
        if not pRoot:
            return True
        left_height = self.FindHeight(pRoot.left)
        right_height = self.FindHeight(pRoot.right)
        return abs(left_height - right_height) <= 1 and self.IsBalanced_Solution(pRoot.left) and self.IsBalanced_Solution(pRoot.right)
```
平衡二叉树：它是一棵空树或它的左右两个子树的高度差的绝对值不超过1，并且左右两个子树都是一棵平衡二叉树<br>
思路：根据定义，从根节点开始，求其左右子树的高度,返回高度差，并把子树递归判断。这样做，会导致额外的计算，比如对子树重复遍历：在求左右子树的高度的时候遍历一次，再判断左右子树是否是平衡二叉树的时候又遍历一次，所以可以考虑在计算子树高度的时候就判断子树是否是平衡二叉树，如果子树不是平衡树，返回-1；如果是的话，返回高度。<br>
```python
# 优化后：从下往上
# -*- coding:utf-8 -*-
class Solution:
    def FindDepth(self, node):
        if not node:
            return 0
        
        left = self.FindDepth(node.left)
        if left == -1:  # 加这个判断是为了提早结束，比如发现左子树不是平衡二叉树的时候，就结束比较
            return -1
        
        right = self.FindDepth(node.right)
        if right == -1:
            return -1
        
        if abs(left-right) > 1:
            return -1
        else:
            return max(left, right) + 1
    
    def IsBalanced_Solution(self, pRoot):
        # write code here
        return self.FindDepth(pRoot) != -1  
```
* 剑指Offer（五十七）：二叉树的下一个结点
```python
# -*- coding:utf-8 -*-
# class TreeLinkNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
#         self.next = None
class Solution:
    def GetNext(self, pNode):
        if pNode.right: 
            cur = pNode.right
            while cur.left:
                cur = cur.left
            return cur
        else:
            while pNode.next:
                if pNode.next.left == pNode:
                    return pNode.next
                pNode = pNode.next
            return None
```
next 指针指向父节点
思路： 
  1. 如果该节点有右子树，就返回右子树的最左边的叶子节点
  2. 如果没有右子树，判断该节点是父节点的左孩子，是的话，返回父节点； 不是的话，就沿着父节点向上查找，直到找到一个节点是它父亲的左孩子，返回父亲节点。情况可以合并。  没有这样的节点的时候，返回空。

* 剑指Offer（五十八）：对称的二叉树
```python
# -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    def fun(self, node1, node2):
        if not node1 and not node2:
            return True
        if (node1 and not node2) or (not node1 and node2):
            return False
        if node1.val == node2.val:
            return self.fun(node1.left, node2.right) and self.fun(node1.right, node2.left)
        else:
            return False
    def isSymmetrical(self, pRoot):
        # write code here
        if not pRoot:
            return True
        return self.fun(pRoot.left, pRoot.right)
```
左右子树分别比较，注意几种特殊情况

* 剑指Offer（五十九）：按之字顺序打印二叉树
```python
import collections
class Solution:
    def Print(self, pRoot):
        # write code here
        if not pRoot:
            return []
        q = collections.deque()
        q.append(pRoot)
        res = []
        flag = True
        while q:
            size = len(q)
            temp = []
            for i in range(size):
                node = q.popleft()
                temp.append(node.val)
                if node.left:
                    q.append(node.left)
                if node.right:
                    q.append(node.right)
            if not flag:
                temp.reverse()
            res.append(temp)
            flag = not flag
        return res
```
同下一题，只是每行打印的时候，用一个flag判断方向<br>

* 剑指Offer（六十）：把二叉树打印成多行
```python
import collections
class Solution:
    # 返回二维列表[[1,2],[4,5]]
    def Print(self, pRoot):
        # write code here
        if not pRoot:
            return []
        q = collections.deque()
        q.append(pRoot)
        res = []
        while q:
            size = len(q)
            temp = []
            for i in range(size):
                node = q.popleft()
                temp.append(node.val)
                if node.left:
                    q.append(node.left)
                if node.right:
                    q.append(node.right)
            res.append(temp)
        return res
```
打印的时候，用长度去限制打印同一行

* 剑指Offer（六十一）：序列化二叉树
```python
class Solution:
    def __init__(self):
        self.serial = []
    def Serialize(self, root):
        # write code here
        if not root:
            self.serial.append('#')
        else:
            self.serial.append(str(root.val))
            self.Serialize(root.left)
            self.Serialize(root.right)
        return ' '.join(self.serial)
    
    def Deserialize(self, s):
        serial = s.split()
        def dePre():
            val = serial.pop(0)
            if val == "#":
                return None
            node = TreeNode(int(val))
            node.left = dePre()
            node.right = dePre()
            return node
        return dePre()
```
没太看懂这个split是在干啥的，没有样例，不知道到底是要搞成啥样？

## 二叉搜索树
剑指Offer（二十三）：二叉搜索树的后序遍历序列
```python
# -*- coding:utf-8 -*-
   
class Solution:
    def helper(self, sequence):
        if len(sequence) < 2:
            return True
        i = 0
        left, right = [], []
        while i < len(sequence) and sequence[i] < sequence[-1]:
            left.append(sequence[i])
            i += 1
        right = sequence[i:-1]
        for j in range(len(right)):
            if right[j] < sequence[-1]:
                return False
        return self.helper(left) and self.helper(right)
    
    def VerifySquenceOfBST(self, sequence):
        # write code here
        if not sequence:
            return False
        return self.helper(sequence)
    
```
思路： 对于 BST，后序遍历的最后一个数字是根节点，BST 左子树的所有节点都比根节点小，右子树的所有节点都比根节点大，因此对前面的元素与根节点大小比较，可以把该序列分成左、右子树两部分。然后再递归判断左右子树的结构是否符合。当有元素违背 BST 的规则时，则返回 False。  当遇到第一个比 sequence[-1] 更大的数时就将数组分成左右两个部分。

剑指Offer（二十六）：二叉搜索树与双向链表
```python
class Solution:
    def Convert(self, pRootOfTree):
        # 中序遍历，修改指针的指向
        if not pRootOfTree:
            return None
        stack = []
        pre, cur = None, pRootOfTree
        isfirst = True
        while cur or stack:
            while cur:
                stack.append(cur)
                cur = cur.left
            if stack:
                cur = stack.pop()
                if isfirst: # 访问到了最左边，是整个链表的起始节点，用root标记，固定住，直接返回
                    root = cur
                    isfirst = False
                else:
                    pre.right = cur
                    cur.left = pre
                pre = cur
                cur = cur.right
        return root
```
非递归版：核心还是中序遍历，在遍历的时候，修改指针的指向，需要保留一个pre指针指向cur的前面一个

```python
class Solution:
    def __init__(self):
        self.pHead = None
        self.pEnd = None  
    def Convert(self, pRootOfTree):
        if not pRootOfTree:
            return None
        self.Convert(pRootOfTree.left)
        if not self.pHead:  # 第一个节点
            self.pHead = pRootOfTree
            self.pEnd = pRootOfTree
        else:
            self.pEnd.right = pRootOfTree
            pRootOfTree.left = self.pEnd
            self.pEnd = pRootOfTree
        self.Convert(pRootOfTree.right)
        
        return self.pHead
```
递归版：
对于 BST，其中序遍历的结果是有序数组，转为排序的双向链表后，某个节点的 left 指向左子树中最大的节点，right 指向右子树中最小的节点。 遍历到某个节点时，它的左边已经排好序了，并且处于链表中的最后一个节点就是该节点的 left，然后再去转换该节点的右子树，其转换规则与左子树一样，因此递归转换即可。

设置两个指针 pHead 和 pEnd 分别指向转换后的双向链表的表头和表尾，首先找到链表的第一个节点，也就是 BST 最左边的节点，此时 pHead 和 pEnd 都还是 None，因此把 pHead 和 pEnd 都指向第一个节点，当遍历到下一个节点时，就需要调整指针的指向，链表中最后一个节点 pEnd 的下一个节点应该是当前节点，当左子树转化完成后，就对右子树进行相同的转换。 整个过程就相当于中序遍历，只是中间改成了指针的调整

剑指Offer（六十二）：二叉搜索树的第k个结点
```python
# -*- coding:utf-8 -*-
class Solution:
    # 返回对应节点TreeNode
    def KthNode(self, pRoot, k):
        # 中序遍历就是排好序的，遍历到第k个停止
        if k <= 0:
            return None
        i = 0
        stack = []
        while pRoot or stack:
            while pRoot:
                stack.append(pRoot)
                pRoot = pRoot.left
            if stack:
                node = stack.pop()
                i += 1
                if i == k:
                    return node
                pRoot = node.right

```
中序遍历，遍历到第k个的时候就停止，注意返回的是节点，而不是节点的值

## 数组
* 剑指Offer（一）：二维数组中的查找
```python
# -*- coding:utf-8 -*-
class Solution:
    # array 二维列表
    def Find(self, target, array):
        # write code here
        i, j = len(array)-1, 0
        while i >= 0 and j < len(array[0]):
            if target > array[i][j]:
                j += 1
            elif target < array[i][j]:
                i -= 1
            else:
                return True
        return False
```
思路：从左下角或者右上角开始，如果从左下角开始，当前值小于target，说明j要右移；大于的话，i要上移动

* 剑指Offer（六）：旋转数组的最小数字
class Solution:
    def minNumberInRotateArray(self, rotateArray):
        if not rotateArray:
            return 0
        left, right = 0, len(rotateArray)-1
        while left < right:
            if rotateArray[left] < rotateArray[right]:
                return rotateArray[left]
            mid = (left + right) // 2
            if rotateArray[mid] > rotateArray[left]:  # 左边有序，最小数字在右边
                left = mid + 1
            elif rotateArray[mid] < rotateArray[right]:  # 右边有序，最小数字在左边，可能是mid
                right = mid
            else:
                left += 1  # left >= mid >= right
        return rotateArray[left]
思路：二分查找，时间复杂度 O(logn)
  1. left < right：说明序列不是旋转，是递增的，直接返回left值；
  2. left < mid： 说明左边有序，小的数字在右边，mid肯定不是
  3. mid < rigt: 说明右边有序，小的数字在左边，mid可能是
  4. left >= mid >= right ： left右移，缩小范围

剑指Offer（十三）：调整数组顺序使奇数位于偶数前面
```python
# -*- coding:utf-8 -*-
class Solution:
    def reOrderArray(self, array):
        odd, even = [], []
        for i in range(len(array)):
            if array[i] % 2 == 1:
                odd.append(array[i])
            else:
                even.append(array[i])
        return odd + even

```
解法一： 空间换时间，开辟两个奇偶辅助数组，边遍历，边存储

```python
class Solution:
    def reOrderArray(self, array):
        for i in range(len(array)):
            for j in range(len(array)-i-1):
                if array[j] % 2 == 0 and array[j+1] % 2 == 1:
                    array[j], array[j+1] = array[j+1], array[j]
        return array
```
解法二：用冒泡排序的思想，因为冒泡排序是稳定的，每一趟将当前序列中 最后一个偶数冒泡到当前最后一个位置。

* 剑指Offer（二十八）：数组中出现次数超过一半的数字
```python
# -*- coding:utf-8 -*-
class Solution:
    def MoreThanHalfNum_Solution(self, numbers):
        # write code here
        d = {}
        for i in range(len(numbers)):
            if numbers[i] in d:
                d[numbers[i]] += 1
            else:
                d[numbers[i]] = 1
        for key, value in d.items():
            if value > len(numbers)// 2:
                return key
        return 0
```
思路：边遍历边计数


剑指Offer（三十）：连续子数组的最大和
```python
class Solution:
    def FindGreatestSumOfSubArray(self, array):
        # write code here
        size = len(array)
        dp = [0 for i in range(size)]
        dp[0] = array[0]
        res = float('-inf')
        for i in range(size):
            dp[i] = max(array[i], dp[i-1]+array[i])
            res = max(res, dp[i])
        return res
```
dp[i]表示以i结尾的连续子数组的最大长度

剑指Offer（三十二）：把数组排成最小的数
```python
# -*- coding:utf-8 -*-
import functools  # py3需要
class Solution:
    def cmp(self, x, y):  # x排在y前面，要return -1
        if x+y > y+x:
            return 1
        if x+y < y+x:
            return -1
        return 0
    def PrintMinNumber(self, numbers):
        # write code here
        numbers = [str(i) for i in numbers]
        numbers = sorted(numbers, self.cmp) # py3写法是：numbers = sorted(numbers, key = functools.cmp_to_key(cmp))
        return ''.join(numbers)
```
思路： leetcode 179，自定义排序规则，数组根据这个规则排序后能排成一个最小的数字。对于两个数 n 和 m，将两个数组合起来，我们需要比较的是 nm 和 mn 哪个更小，如果 nm < mn，那么 n 应该排在前面. 上面这种写法是py2的，对于py3变了，key只能接收一个参数，没有cmp的方法，所以不能直接用，需要导包。 <br>

注意：str 与 list 互相转换： <br>
str -> list： list(s) <br>
list -> str: ''.join(l) 要求list中的每个元素必须是str类型的，如果是int要先转换  <br>

剑指Offer（三十五）：数组中的逆序对
```python
# -*- coding:utf-8 -*-
class Solution:
    def __init__(self):
        self.cnt = 0
    
    def InversePairs(self, data):
        if len(data) <= 1:
            return 0
        self.helper(data)
        return self.cnt % 1000000007
          
    def helper(self, data):
        if len(data) <= 1:
            return data
        mid = len(data) // 2
        left = self.helper(data[:mid])
        right = self.helper(data[mid:])
        i, j = 0, 0
        tmp = []
        while i < len(left) and j < len(right):
            if left[i] <= right[j]:
                tmp.append(left[i])
                i += 1
            else:
                tmp.append(right[j])
                j += 1
                self.cnt = self.cnt + (len(left) - i)
        if i < len(left):
            tmp += left[i:]
        else:
            tmp += right[j:]
        return tmp
```
思路：如果固定一个数字，让它与后面所有的数字作比较，时间复杂度为 O(n^2)，超时，因此考虑只比较相邻的两个数字。

可以把问题分解，先将数组中所有的数字拆分为长度为 1 的子数组，再将相邻的两个子数组进行比较，统计出逆序对的个数，同时对两个子数组按照从小到大的顺序进行排序和合并。

用两个指针 i 和 j 分别指向两个子数组的第一个数字，并比较指向数字的大小。如 left=[5,7] 和 right=[4,6]
若第一个子数组中的数字（left[i]=5）大于第二个子数组中的数字（right[j]=4），那么第一个子数组中的包括 i 和 i 之后的所有数字都会大于 right[j] 这个数字，并构成逆序对的个数为 len(left)-i 个，并将 j+1；若小于或等于，则不构成逆序对。
开辟一个辅助数组 tmp，每次把比较中较小的数字添加到 tmp 中，确保其是从小到大排序的。
以上思想类似于归并排序： 先将数组拆分为长度为 1 的子数组，两两合并子数组并统计出相邻两个子数组之间逆序对的个数，合并过程中需要对两个子数组进行排序。

剑指Offer（三十七）：数字在排序数组中出现的次数

剑指Offer（四十）：数组中只出现一次的数字
剑指Offer（五十）：数组中重复的数字
剑指Offer（五十一）：构建乘积数组

## 字符串
剑指Offer(二)：替换空格
剑指Offer（二十七）：字符串的排列
剑指Offer（三十四）：第一个只出现一次的字符
剑指Offer（四十三）：左旋转字符串
剑指Offer（四十四）：翻转单词顺序序列
剑指Offer（四十九）：把字符串转换成整数
剑指Offer（五十二）：正则表达式匹配
剑指Offer（五十三）：表示数值的字符串


## 栈
剑指Offer（五）：用两个栈实现队列
```python
# -*- coding:utf-8 -*-
class Solution:
    def __init__(self):
        self.stack1 = []
        self.stack2 = []
        
    def push(self, node):
        self.stack1.append(node)
        return self.stack1[-1]
    
    def pop(self):
        if self.stack2:
            return self.stack2.pop()
        while self.stack1:
            self.stack2.append(self.stack1.pop())
        return self.stack2.pop()
```
思路： 入队：直接追加到stack1的最后，并返回最后一个元素；<br>
出队：如果stack2中有元素，直接弹出；没有的话，把stack1中的所有元素都压入stack2中，再弹出顶上的元素<br>
延伸的选择题：要是stack1的容量是 m ， stack2的容量是 n，m > n, 问能模拟的队列最长是：2 * n + 1

剑指Offer（二十）：包含min函数的栈
剑指Offer（二十一）：栈的压入、弹出序列

## 递归
剑指Offer（七）：裴波那契数列
```python
class Solution:
    def Fibonacci(self, n):
        # write code here
        if n == 0:
            return 0
        if n == 1:
            return 1
        a, b = 0, 1
        for i in range(2,n+1):
            res = a+ b
            a = b
            b = res
        return res
```
用a和b这两个临时变量保存

剑指Offer（八）：跳台阶
```python
class Solution:
    def jumpFloor(self, number):
        # write code here
        if number == 0:
            return 1
        if number == 1:
            return 1
        a, b = 1, 1
        for i in range(2, number+1):
            res = a + b
            a = b
            b = res
        return res
```
同上，只不过0台阶，返回的是1

剑指Offer（九）：变态跳台阶
```python
class Solution:
    def jumpFloorII(self, number):
        # write code here
        if number == 0:
            return 0
        if number == 1:
            return 1
        a = 1
        for i in range(2, number+1):
            res = 2 * a
            a = res
        return res
```
F(n) = 2 * F(n-1) n >=2, n==1时, F(1) =1

剑指Offer（十）：矩形覆盖
```python
class Solution:
    def rectCover(self, number):
        # write code here
        if number == 1 or number == 0:
            return number
        a, b = 1, 1
        for i in range(2, number+1):
            res = a + b
            a = b
            b = res
        return res
```
思路：依然是菲波那切数列，只不过number=0的时候，要返回0；但在后续计算的时候，把0这个位置赋为1 F(n) = F(n-1) + F(n-2)

## 回溯法
剑指Offer（六十五）：矩阵中的路径
剑指Offer（六十六）：机器人的运动范围

## 其他
* 剑指Offer（二十九）：最小的K个数，要求输出的k个数是有序的
```python
class Solution:
    def GetLeastNumbers_Solution(self, tinput, k):
        # write code here
        if not tinput or k <= 0 or k > len(tinput):
            return []
        tinput.sort()
        return tinput[:k]
```
思路一：整个数组排序，输出前K个数
```python
# -*- coding:utf-8 -*-
class Solution:
    # 构建一个最大堆
    def ModifyMaxHeap(self, heap, root):
        left = 2 * root + 1
        right = 2 * root + 2
        largest = root
        if left < len(heap) and heap[left] > heap[largest]:
            largest = left
        if right < len(heap) and heap[right] > heap[largest]:
            largest = right
        if largest != root:
            heap[largest], heap[root] = heap[root], heap[largest]
            self.ModifyMaxHeap(heap, largest)
            
    def BuildMaxHeap(self, heap):
        size = len(heap)
        for i in range(size-1, -1, -1):
            self.ModifyMaxHeap(heap, i)
            
    def TopK(self, tinput, k):
        size = len(tinput)
        heap = tinput[:k]
        self.BuildMaxHeap(heap)  # 构建一个容量为k的最大堆
        for i in range(k, size):
            if tinput[i] < heap[0]: # heap[0]始终是堆中最大值
                tinput[i], heap[0] = heap[0], tinput[i]
                self.ModifyMaxHeap(heap, 0)
        return sorted(heap) 
    
    def GetLeastNumbers_Solution(self, tinput, k):
        # write code here
        if not tinput or k <= 0 or k > len(tinput):
            return []
        return self.TopK(tinput, k)
```
思路二：创建一个大小为 k 的容器来存储最小的 k 个数，遍历数组 <br>
若容器中的数字少于 k 个，则直接把读入的数字加入容器中<br>
若容器中的数字等于 k 个，则不能再插入数字，只能替换已有的数字，先找到容器中最大的数，然后与新数字进行比较，若待插入的数字比当前最大值小，则替换当前最大值；若待插入的数字比当前最大值还大，那么它不可能成为最小的 k 个数，舍弃。<br>
我们可以用二叉树来实现这个容器，在 O(logk) 的时间里能完成一次操作，总共有 n 个数字，则时间复杂度为 O(nlogk)  <br>

由于每次都要去找容器中的最大值，我们可以用最大堆来实现，最大堆中根节点的值是最大的，那么找到每次找到容器中最大值的时间复杂度为 O(1)，同时需要 O(logk) 的时间来完成删除和插入新节点。<br>

剑指Offer（十一）：二进制中1的个数
剑指Offer（十二）：数值的整数次方
剑指Offer（十九）：顺时针打印矩阵
剑指Offer（二十九）：最小的K个数
剑指Offer（三十一）：整数中1出现的次数（从1到n整数中1出现的次数）
剑指Offer（三十三）：丑数
剑指Offer（四十一）：和为S的连续正数序列
剑指Offer（四十二）：和为S的两个数字
剑指Offer（四十五）：扑克牌顺子
剑指Offer（四十六）：孩子们的游戏（圆圈中最后剩下的数）
剑指Offer（四十七）：求1+2+3+…+n
剑指Offer（四十八）：不用加减乘除的加法
剑指Offer（五十四）：字符流中第一个不重复的字符
剑指Offer（六十三）：数据流中的中位数
剑指Offer（六十四）：滑动窗口的最大值
