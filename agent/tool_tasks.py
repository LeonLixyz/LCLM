"""Fresh, deterministic tool-use diagnostics with executable reference outcomes.

This is a custom diagnostic, not a BFCL/ToolBench leaderboard implementation.
Only initial_messages() and the public tool schemas are exposed to the model.
"""
from __future__ import annotations

import copy
import hashlib
import json
import random
import re

VERSION = 'native-tools-v1'
DEFAULT_SEED = 2026091601
FAMILIES = ('direct', 'chain', 'recovery', 'memory')


def tool(name, description, **properties):
    return {'type': 'function', 'function': {
        'name': name, 'description': description,
        'parameters': {'type': 'object', 'properties': {
            key: {'type': kind} for key, kind in properties.items()},
            'required': list(properties), 'additionalProperties': False}}}


TOOLS = {
    'inventory_lookup': tool('inventory_lookup', 'Look up an item code: serial, price in cents, stock and current version.', item_code='string'),
    'resolve_item': tool('resolve_item', 'Resolve a human-readable item alias to its item code.', alias='string'),
    'multiply': tool('multiply', 'Multiply two integers exactly.', a='integer', b='integer'),
    'weather_lookup': tool('weather_lookup', 'Get the synthetic weather for a city.', city='string'),
    'reserve': tool('reserve', 'Reserve units of an item using its current inventory version. Returns a receipt and remaining stock. A repeat reservation is rejected.', item_code='string', quantity='integer', version='integer'),
    'expand': tool('expand', 'Return the original uncompressed text of a context segment, such as seg_3.', segment_id='string'),
}
TOOLS['expand']['function']['strict'] = True
TOOLS['expand']['function']['parameters']['properties']['segment_id']['pattern'] = r'^seg_[1-9][0-9]*$'


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def call(name, **arguments):
    return {'name': name, 'arguments': arguments}


def make_tasks(per_family=50, seed=DEFAULT_SEED):
    if per_family < 1:
        raise ValueError('per_family must be positive')
    tasks = []
    for family in FAMILIES:
        for index in range(per_family):
            task_id = f'{VERSION}-{seed}-{family}-{index:04d}'
            rng = random.Random(task_id)
            code = lambda prefix: f'{prefix}-{rng.getrandbits(48):012x}'
            item = code('ITEM')
            alias = code('part')
            record = {'serial': code('SER'), 'price_cents': rng.randrange(100, 90000),
                      'stock': rng.randrange(20, 100), 'version': rng.randrange(1, 100)}
            task = dict(task_id=task_id, family=family, subtype=family,
                        inventory={item: record}, aliases={alias: item}, segments={},
                        support=[], expected=None, reference_calls=[], no_tool=False,
                        injected_failure=family == 'recovery', required_tools=[],
                        tools=['inventory_lookup', 'multiply', 'weather_lookup'])
            if family == 'direct':
                subtype = index % 5
                if subtype in (0, 4):
                    field = 'serial' if subtype == 0 else 'price_cents'
                    task.update(subtype=f'lookup_{field}', question=f'Look up {item} and return its {field}.',
                                expected=record[field], required_tools=['inventory_lookup'],
                                reference_calls=[call('inventory_lookup', item_code=item)])
                elif subtype == 1:
                    a, b = rng.randrange(11, 999), rng.randrange(11, 999)
                    task.update(subtype='arithmetic', question=f'Use the multiply tool to calculate {a} times {b}.',
                                expected=a*b, required_tools=['multiply'], reference_calls=[call('multiply', a=a, b=b)])
                elif subtype == 2:
                    second = code('ITEM')
                    task['inventory'][second] = dict(record, serial=code('SER'))
                    task.update(subtype='multiple_calls', question=f'Return the serials for {item} and {second}, as a JSON object mapping item codes to serials.',
                                expected={key: task['inventory'][key]['serial'] for key in (item, second)},
                                required_tools=['inventory_lookup'], reference_calls=[call('inventory_lookup', item_code=key) for key in (item, second)])
                else:
                    literal = code('ACK')
                    task.update(subtype='no_tool', no_tool=True, expected=literal,
                                question=f'Return this exact acknowledgement: {literal}. All information is supplied here; do not call any tool.')
            elif family == 'chain':
                task['tools'] = ['resolve_item', 'inventory_lookup', 'reserve', 'weather_lookup']
                task['required_tools'] = ['resolve_item', 'inventory_lookup']
                task['reference_calls'] = [call('resolve_item', alias=alias), call('inventory_lookup', item_code=item)]
                if index % 2 == 0:
                    task.update(subtype='dependent_lookup', question=f'Find the serial of the item whose alias is {alias}.', expected=record['serial'])
                else:
                    quantity = rng.randrange(1, 10)
                    receipt = 'RSV-' + fingerprint([task_id, item, quantity])[:16]
                    task.update(subtype='stateful_reservation', quantity=quantity, reserve_item=item,
                                question=f'Resolve item alias {alias}, check its current inventory, then reserve exactly {quantity} units. Return an object with receipt and remaining_stock.',
                                expected={'receipt': receipt, 'remaining_stock': record['stock']-quantity})
                    task['required_tools'].append('reserve')
                    task['reference_calls'].append(call('reserve', item_code=item, quantity=quantity, version=record['version']))
            elif family == 'recovery':
                task.update(question=f'Look up the serial for {item}. If the service reports a retryable error, retry the operation.',
                            expected=record['serial'], required_tools=['inventory_lookup'],
                            reference_calls=[call('inventory_lookup', item_code=item)]*2)
            else:
                task['tools'] = ['expand']
                target, vendor = code('ASSET'), code('VENDOR')
                n_segments = 8 if index % 2 == 0 else 16
                support_indices = rng.sample(range(n_segments), 2)
                access_key = code('KEY')
                for j in range(n_segments):
                    entity, value = code('ASSET'), code('KEY')
                    fact = f'Asset {entity} has access key {value}.'
                    if j == support_indices[0]:
                        fact = (f'Asset {target} has access key {access_key}.' if index % 2 == 0
                                else f'Asset {target} is assigned to vendor {vendor}.')
                    if index % 2 and j == support_indices[1]:
                        fact = f'Vendor {vendor} has access key {access_key}.'
                    lines = [f'Archive note {k}: batch {code("BATCH")} was reviewed; this note records only routine audit metadata.' for k in range(24)]
                    lines.insert(rng.randrange(len(lines)+1), 'Authoritative record: ' + fact)
                    task['segments'][f'seg_{j+1}'] = '\n'.join(lines)
                support = [f'seg_{j+1}' for j in support_indices[:1 if index % 2 == 0 else 2]]
                task.update(subtype='memory_lookup' if index % 2 == 0 else 'memory_join', support=support,
                            question=f'What is the access key {"of the vendor assigned to " if index % 2 else "for "}asset {target}?',
                            expected=access_key, reference_calls=[call('expand', segment_id=segment) for segment in support])
            task['question'] += '\nFinish with FINAL: {"answer": <your result as a JSON value>}.'
            tasks.append(task)
    return tasks


def initial_messages(task, mode):
    content = task['question']
    if task['segments']:
        blocks = []
        for name, text in task['segments'].items():
            if mode != 'plain':
                text = f'<|memory_start|>{text}<|memory_end|>'
            blocks.append(f'{name}\n{text}')
        content = '\n\n'.join(blocks) + '\n\n' + content
    system = 'Complete the task using the available tools when needed. Use native tool calls. Read tool results before continuing. Finish with the requested FINAL JSON. Do not emit thinking tags.'
    if mode == 'compressed_expand':
        system += ' Numbered seg_i blocks are compressed memories. Use expand to inspect original text when exact details are needed.'
    names = task['tools'] if task['family'] != 'memory' or mode == 'compressed_expand' else []
    return [{'role': 'system', 'content': system}, {'role': 'user', 'content': content}], [copy.deepcopy(TOOLS[name]) for name in names]


class ToolEnvironment:
    """Private, in-memory executor; no shell, networking or real external mutations."""
    def __init__(self, task, tools):
        self.task = task
        self.inventory = copy.deepcopy(task['inventory'])
        self.schemas = {tool['function']['name']: tool['function']['parameters'] for tool in tools}
        self.events = []
        self.reservations = []
        self.injected_errors = 0

    def execute(self, name, args):
        event = {'name': name, 'arguments': copy.deepcopy(args), 'schema_valid': False}
        schema = self.schemas.get(name)
        error = None
        if schema is None:
            error = 'unknown_tool'
        elif not isinstance(args, dict) or set(args) != set(schema['required']):
            error = 'invalid_argument_keys'
        elif any(type(args[k]) is not {'string': str, 'integer': int}[prop['type']] for k, prop in schema['properties'].items()):
            error = 'invalid_argument_type'
        elif any('pattern' in prop and not re.fullmatch(prop['pattern'], args[k])
                 for k, prop in schema['properties'].items()):
            error = 'invalid_argument_pattern'
        if error:
            result = {'error': error}
        else:
            event['schema_valid'] = True
            result = self._execute(name, args)
        event['result'] = copy.deepcopy(result)
        event['success'] = 'error' not in result
        self.events.append(event)
        return result

    def _execute(self, name, args):
        if name == 'inventory_lookup':
            if self.task['injected_failure'] and not self.injected_errors:
                self.injected_errors += 1
                return {'error': 'temporarily_unavailable', 'retryable': True}
            record = self.inventory.get(args['item_code'])
            return copy.deepcopy(record) if record else {'error': 'unknown_item'}
        if name == 'resolve_item':
            code = self.task['aliases'].get(args['alias'])
            return {'item_code': code} if code else {'error': 'unknown_alias'}
        if name == 'multiply':
            if max(abs(args['a']), abs(args['b'])) > 10**9:
                return {'error': 'integer_out_of_range'}
            return {'result': args['a'] * args['b']}
        if name == 'weather_lookup':
            return {'city': args['city'], 'condition': 'clear'}
        if name == 'expand':
            text = self.task['segments'].get(args['segment_id'])
            return {'text': text} if text is not None else {'error': 'unknown_segment'}
        if name == 'reserve':
            item, quantity = args['item_code'], args['quantity']
            record = self.inventory.get(item)
            if not record or quantity < 1 or quantity > record['stock']:
                return {'error': 'invalid_reservation'}
            if self.reservations:
                return {'error': 'already_reserved'}
            if record['version'] != args['version']:
                return {'error': 'stale_version', 'retryable': True}
            record['stock'] -= quantity
            record['version'] += 1
            receipt = 'RSV-' + fingerprint([self.task['task_id'], item, quantity])[:16]
            self.reservations.append({'item_code': item, 'quantity': quantity, 'receipt': receipt})
            return {'receipt': receipt, 'remaining_stock': record['stock']}
        raise AssertionError(f'No executor for schema: {name}')
