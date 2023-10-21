def twoSum(, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        for index,num in enumerate(nums):
            add = target - num
            for indexs,nnum in enumerate(nums):
                if nnum is add and indexs is not index:
                    return index,indexs

if __name__ == '__main__':
     xx = [2222222,2222222]
     target = 4444444
     m = twoSum(xx,target)