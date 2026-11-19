import sys

def run(proto, layout, iterations):
    keys = ['BusRd', 'BusRdX', 'BusUpgr', 'Invalidations',
            'Writebacks', 'CacheToCache', 'Silent']
    stats = dict.fromkeys(keys, 0)
    cache = [{'A': 'I', 'B': 'I'}, {'A': 'I', 'B': 'I'}]

    x, y = ('A', 'A') if layout == 'adjacent' else ('A', 'B')

    def access(core, line, op):
        other = 1 - core
        state = cache[core][line]
        other_state = cache[other][line]

        if op == 'R':
            if state != 'I':
                return
            stats['BusRd'] += 1
            if other_state != 'I':
                if other_state == 'M':
                    if proto == 'MOESI':
                        cache[other][line] = 'O'
                        stats['CacheToCache'] += 1
                    else:
                        cache[other][line] = 'S'
                        stats['Writebacks'] += 1
                elif other_state == 'E':
                    cache[other][line] = 'S'
                cache[core][line] = 'S'
            else:
                cache[core][line] = 'E' if proto != 'MSI' else 'S'

        else:
            if state == 'M':
                return
            if state == 'E':
                cache[core][line] = 'M'
                stats['Silent'] += 1
                return
            if state == 'S':
                stats['BusUpgr'] += 1
            else:
                stats['BusRdX'] += 1

            if other_state != 'I':
                stats['Invalidations'] += 1
                cache[other][line] = 'I'
            cache[core][line] = 'M'

    for _ in range(iterations):
        access(0, x, 'R')
        access(0, x, 'W')
        access(1, y, 'R')
        access(1, y, 'W')

    return stats

iterations = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
print("CACHE COHERENCE SIMULATION")
print("Iterations:", iterations)

for layout in ['adjacent', 'padded']:
    print("\nLayout:", layout)
    for proto in ['MSI', 'MESI', 'MOESI']:
        s = run(proto, layout, iterations)
        print(proto, s)
