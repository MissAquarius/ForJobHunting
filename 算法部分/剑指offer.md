## 链表
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

```
先让长的链表把头部多出来的部分"砍掉"，然后两个指针再一起遍历，跟上面的思路大概一样。
## 二叉树

## 二叉搜索树

## 数组

## 字符串

## 栈

## 递归

## 回溯法

## 其他
