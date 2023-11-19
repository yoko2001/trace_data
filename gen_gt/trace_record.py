def decorator(cls):
    # 给被装饰的类添加一个__getitem__方法
    class MyCls(cls):
        def __getitem__(self, item):
            return getattr(self, item)

    return MyCls

@decorator
class TraceRecord(object):
    # 限制属性
    __slots__ = [
        'process', 'se', 'timestamp', 
        'dir', 'swap_level', 'left', 
        'entry', 'va', 'folio',
        'swapprio_b', 'readahead_b', 'gen',
        'memcg_id', 'minseq', 'ref', 
        'tier', 'se_hist', 'se_ts'
    ]

    def __init__(
        self,  
        process, se, timestamp, dir, 
        swap_level, left, entry, va, 
        folio, swapprio_b, readahead_b, gen,
        memcg_id, minseq, ref, tier, se_hist, 
        se_ts
    ):
        self.process = process
        self.se = se
        self.timestamp = timestamp
        self.dir = dir
        self.swap_level = swap_level
        self.left = left
        self.entry = entry
        self.va = va
        self.folio = folio
        self.swapprio_b = swapprio_b
        self.readahead_b = readahead_b
        self.gen = gen
        self.memcg_id = memcg_id
        self.minseq = minseq
        self.ref = ref
        self.tier = tier
        self.se_hist = se_hist
        self.se_ts = se_ts

    def __setitem__(self, key, value):
        if key == 'title':
            if isinstance(value, str):
                super().__setattr__(key, value)
            else:
                raise TypeError('1title 只能设置为字符串，不能设置为其他类型', value)
    
    def __setattr__(self, key, value):
        if key == 'process':
            if isinstance(value, str):
                super().__setattr__(key, value)
            else:
                raise TypeError('process 只能设置为str类型，不能设置为其他类型')
        elif key == 'se':
            if isinstance(value, int):
                super().__setattr__(key, value)
            else:
                raise TypeError('se 只能设置为int类型，不能设置为其他类型')
        elif key == 'timestamp':
            if isinstance(value, float):
                super().__setattr__(key, value)
            else:
                raise TypeError('timestamp 只能设置为float类型，不能设置为其他类型')
        elif key == 'dir':
            if isinstance(value, str):
                super().__setattr__(key, value)
            else:
                raise TypeError('dir 只能设置为str类型，不能设置为其他类型')
        elif key == 'swap_level':
            if isinstance(value, str):
                super().__setattr__(key, value)
            else:
                raise TypeError('swap_level 只能设置为str类型，不能设置为其他类型')
        elif key == 'left':
            if isinstance(value, int):
                super().__setattr__(key, value)
            else:
                raise TypeError('left 只能设置为int类型，不能设置为其他类型')
        elif key == 'entry':
            if isinstance(value, int):
                super().__setattr__(key, value)
            else:
                raise TypeError('entry 只能设置为int类型，不能设置为其他类型')
        elif key == 'va':
            if isinstance(value, int):
                super().__setattr__(key, value)
            else:
                raise TypeError('va 只能设置为int类型，不能设置为其他类型')
        elif key == 'folio':
            if isinstance(value, int):
                super().__setattr__(key, value)
            else:
                raise TypeError('folio 只能设置为int类型，不能设置为其他类型')
        elif key == 'swapprio_b':
            if isinstance(value, str):
                super().__setattr__(key, value)
            else:
                raise TypeError('swapprio_b 只能设置为str类型，不能设置为其他类型')
        elif key == 'readahead_b':
            if isinstance(value, int):
                super().__setattr__(key, value)
            else:
                raise TypeError('readahead_b 只能设置为int类型，不能设置为其他类型')
        elif key == 'gen':
            if isinstance(value, int):
                super().__setattr__(key, value)
            else:
                raise TypeError('gen 只能设置为int类型，不能设置为其他类型')
        elif key == 'memcg_id':
            if isinstance(value, int):
                super().__setattr__(key, value)
            else:
                raise TypeError('memcg_id 只能设置为int类型，不能设置为其他类型')
        elif key == 'ref':
            if isinstance(value, int):
                super().__setattr__(key, value)
            else:
                raise TypeError('ref 只能设置为int类型，不能设置为其他类型')
        elif key == 'tier':
            if isinstance(value, int):
                super().__setattr__(key, value)
            else:
                raise TypeError('tier 只能设置为int类型，不能设置为其他类型')
        elif key == 'se_hist':
            if value == None:
                pass
            else:
                if isinstance(value, list) and (3 == len(value)):
                    super().__setattr__(key, value)
                else:
                    print(type(value))
                    raise TypeError('se_hist 只能设置为list(len=3)类型，不能设置为其他类型')
        elif key == 'se_ts':
            if value == None:
                pass
            else:
                if isinstance(value, int):
                    super().__setattr__(key, value)
                else:
                    raise TypeError('se_ts 只能设置为int类型，不能设置为其他类型')
        elif key == 'minseq':
            if isinstance(value, int):
                super().__setattr__(key, value)
            else:
                raise TypeError('minseq 只能设置为int类型，不能设置为其他类型')
        else:
            super().__setattr__(key, value)

    def __delattr__(self, item):
        # 判断是否为name属性
        if item == 'entry':
            raise AttributeError('entry属性不能被删除')
        else:
            super().__delattr__(item)

    def __getattr__(self, item):
        # 判断是否为money属性
        if item == 'money':
            value = 0
            return value
    # #
    # def __getitem__(self, item):
    #     return getattr(self, item)

def load_str_record(line_str):
    line_list = eval(line_str)
    assert(len(line_list) == 16 or len(line_list) == 20)
    if (len(line_list)) == 16:
        se = (line_list[1] == "folio_ws_chg_se")
        return TraceRecord( 
#['pagewalker-1880', 'folio_ws_chg', 474.336338, 'r', 's', -2, '8a550', '7f6429a45', 'ced70731360', 'm', '0', '-1', '60', 437, 0, 0]
            process=str(line_list[0]), 
            se=se, 
            timestamp=(float)(line_list[2]), 
            dir=str(line_list[3]), 
            swap_level=str(line_list[4]), 
            left = int(line_list[5]), 
            entry = int(str(line_list[6]), 16), 
            va = int(str(line_list[7]), 16),
            folio = int(str(line_list[8]), 16),
            swapprio_b = str(line_list[9]),
            readahead_b = int(line_list[10]),
            gen = int(line_list[11]),
            memcg_id = int(line_list[12]),
            minseq = int(line_list[13]),
            ref = int(line_list[14]),
            tier = int(line_list[15]),
            se_hist = None,
            se_ts = None
        )
    elif len(line_list) == 20:
        se = (line_list[1] == "folio_ws_chg_se")
        return TraceRecord( 
            process=str(line_list[0]), 
            se=se, 
            timestamp=(float)(line_list[2]), 
            dir=str(line_list[3]), 
            swap_level=str(line_list[4]), 
            left = int(line_list[5]), 
            entry = int(str(line_list[6]), 16), 
            va = int(str(line_list[7]), 16),
            folio = int(str(line_list[8]), 16),
            swapprio_b = str(line_list[9]),
            readahead_b = int(line_list[10]),
            gen = int(line_list[11]),
            memcg_id = int(line_list[12]),
            minseq = int(line_list[13]),
            ref = int(line_list[14]),
            tier = int(line_list[15]),
            se_ts = int(line_list[16]),
            se_hist = [int(line_list[17]),int(line_list[18]),int(line_list[19])]
        )
    else:
        print("err")
        return None
#['pagewalker-1880', 'folio_ws_chg_se', 217.439755, 'e', 'f', 16319, '400000000000203', '0', 'ced7070e2c5', 'm', '0', '-1', '60', 3, 0, 0]

if __name__ == '__main__':
    m = TraceRecord( 
        process='pagewalker-1880', 
        se=1, 
        timestamp=(float)(217.439755), 
        dir='e', swap_level='f', 
        left = 16319, 
        entry = int("400000000000203", 16), 
        va = int("0", 16),
        folio = int("ced7070e2c5", 16),
        swapprio_b = 'm',
        readahead_b = 0,
        gen = -1,
        memcg_id = 60,
        minseq = 309,
        ref = 1,
        tier = 2,
        se_hist = [234, 345, 567],
        se_ts = 123
    )

    print(
        m["process"], #pid
        m["se"], 
        m["timestamp"], 
        m['dir'], # in or out
        m['swap_level'], # fast or slow
        m['left'], # space left in swap_level device
        m['entry'], # page id
        m['va'], 
        m['folio'], 
        m['swapprio_b'],
        m['readahead_b'],
        m['gen'],
        m['memcg_id'],
        m['minseq'],
        m['ref'],
        m['tier'],
        m['se_hist'],
        m['se_ts']
    )