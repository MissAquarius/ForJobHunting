## DFS
* 图示：
![image](https://github.com/MissAquarius/ForJobHunting/blob/master/image/DFS.png)


## BFS
* 常用于：求最短路径、至少需要几步的问题
* 搜索过程（借助队列和一个一位数组实现）：
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
* 
