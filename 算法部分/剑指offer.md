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

```
思路：暂时没看懂

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



## 二叉搜索树

## 数组

## 字符串

## 栈

## 递归

## 回溯法

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
