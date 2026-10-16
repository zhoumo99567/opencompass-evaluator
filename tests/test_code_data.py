import hashlib
import io
import json
import os
from pathlib import Path
import stat
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import code_data as data


def jsonl(rows):
    return (''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows)).encode()


def mbpp_fixture():
    return jsonl([dict(task_id=index, text='Write a function.', code='def f(): return 1',
                       test_setup_code='', test_list=['assert f() == 1'])
                  for index in range(1, 975)])


def cn_fixture(chinese=True):
    return jsonl([dict(task_id=f'HumanEval/{index}', prompt='返回一。' if chinese else 'Return one.',
                       test='def check(candidate): assert candidate() == 1', entry_point='f')
                  for index in range(164)])


def zip_fixture(members):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in members:
            archive.writestr(name, content)
    return stream.getvalue()


class Response(io.BytesIO):
    def __init__(self, content, headers=None):
        super().__init__(content)
        self.headers = headers or {}


class CodeDataTests(unittest.TestCase):
    def test_missing_cn_is_explicit_and_never_replaced_with_english(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = data.code_data_status(tmp)
            self.assertFalse(state['humaneval_cn']['ready'])
            self.assertIn('尚无已核验', state['humaneval_cn']['reason'])
            path = Path(tmp) / data.DATA_FILES['humaneval_cn']
            path.parent.mkdir(parents=True)
            path.write_bytes(cn_fixture(False))
            state = data.code_data_status(tmp)
            self.assertFalse(state['humaneval_cn']['ready'])
            self.assertIn('不能将英文', state['humaneval_cn']['reason'])

    def test_valid_cn_schema_and_duplicate_ids(self):
        self.assertEqual(data._validate_data('humaneval_cn', cn_fixture()), 164)
        rows = data._jsonl_records(cn_fixture())
        rows[-1]['task_id'] = rows[0]['task_id']
        with self.assertRaisesRegex(data.CodeDataError, '编号'):
            data._validate_data('humaneval_cn', jsonl(rows))
        rows[-1]['task_id'] = 'HumanEval/163'
        rows[-1].pop('test')
        with self.assertRaisesRegex(data.CodeDataError, '缺少'):
            data._validate_data('humaneval_cn', jsonl(rows))

    def test_invalid_json_and_oversized_file_are_safe_errors(self):
        for content in (b'{bad}', b'[1]\n', b'\xff'):
            with self.assertRaises(data.CodeDataError):
                data._jsonl_records(content)
        with patch.object(data, 'MAX_JSONL_BYTES', 5):
            with self.assertRaisesRegex(data.CodeDataError, '大小上限'):
                data._jsonl_records(b'123456')

    def test_mbpp_pins_jsonl_even_when_schema_is_valid(self):
        content = mbpp_fixture()
        with self.assertRaisesRegex(data.CodeDataError, '内容校验失败'):
            data._validate_data('mbpp', content)
        with patch.object(data, 'MBPP_JSONL_SHA256', hashlib.sha256(content).hexdigest()):
            self.assertEqual(data._validate_data('mbpp', content), 974)
            rows = data._jsonl_records(content)
            rows[-1]['task_id'] = rows[0]['task_id']
            with self.assertRaisesRegex(data.CodeDataError, '编号'):
                data._validate_data('mbpp', jsonl(rows))

    def test_zip_only_reads_known_member_and_rejects_traversal(self):
        content = mbpp_fixture()
        with patch.object(data, 'MBPP_JSONL_SHA256', hashlib.sha256(content).hexdigest()):
            archive = zip_fixture([(data.MBPP_MEMBER, content), ('mbpp/extra.txt', b'not extracted')])
            self.assertEqual(data._mbpp_from_archive(archive), content)
            for name in ('../escape', '/absolute', 'C:/drive', 'mbpp\\escape'):
                archive = zip_fixture([(data.MBPP_MEMBER, content), (name, b'x')])
                with self.assertRaisesRegex(data.CodeDataError, '不安全'):
                    data._mbpp_from_archive(archive)

    def test_zip_rejects_symlink_and_bad_zip(self):
        stream = io.BytesIO()
        entry = zipfile.ZipInfo('link')
        entry.create_system = 3
        entry.external_attr = (stat.S_IFLNK | 0o777) << 16
        with zipfile.ZipFile(stream, 'w') as archive:
            archive.writestr(entry, '../target')
        with self.assertRaisesRegex(data.CodeDataError, '不安全'):
            data._mbpp_from_archive(stream.getvalue())
        with self.assertRaisesRegex(data.CodeDataError, '格式无效'):
            data._mbpp_from_archive(b'not zip')

    def test_download_checks_archive_hash_and_byte_limit(self):
        with patch.object(data.urllib.request, 'urlopen', return_value=Response(b'changed')):
            with self.assertRaisesRegex(data.CodeDataError, 'SHA256'):
                data._download_mbpp()
        with patch.object(data, 'MAX_DOWNLOAD_BYTES', 10), \
             patch.object(data.urllib.request, 'urlopen', return_value=Response(b'x' * 11)):
            with self.assertRaisesRegex(data.CodeDataError, '大小上限'):
                data._download_mbpp()
        with patch.object(data.urllib.request, 'urlopen',
                          return_value=Response(b'', {'Content-Length': str(data.MAX_DOWNLOAD_BYTES + 1)})):
            with self.assertRaisesRegex(data.CodeDataError, '大小上限'):
                data._download_mbpp()

    def test_download_source_is_fixed_and_success_requires_both_hashes(self):
        content = mbpp_fixture()
        archive = zip_fixture([(data.MBPP_MEMBER, content)])
        with patch.object(data, 'MBPP_JSONL_SHA256', hashlib.sha256(content).hexdigest()), \
             patch.object(data, 'MBPP_ARCHIVE_SHA256', hashlib.sha256(archive).hexdigest()), \
             patch.object(data.urllib.request, 'urlopen', return_value=Response(archive)) as download:
            self.assertEqual(data._download_mbpp(), content)
            self.assertEqual(download.call_args.args[0].full_url, data.MBPP_URL)

    def test_prepare_is_atomic_private_and_reuses_verified_data(self):
        content = mbpp_fixture()
        logs = []
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(data, 'MBPP_JSONL_SHA256', hashlib.sha256(content).hexdigest()), \
             patch.object(data, '_download_mbpp', return_value=content) as download:
            state = data.prepare_code_data(tmp, logs.append)
            path = Path(tmp) / data.DATA_FILES['mbpp']
            self.assertTrue(state['mbpp']['ready'])
            self.assertFalse(state['humaneval_cn']['ready'])
            self.assertEqual(state['mbpp']['count'], 974)
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
            self.assertEqual(list(path.parent.glob('.code-data-*')), [])
            data.prepare_code_data(tmp)
            download.assert_called_once()
            self.assertTrue(any('尚无已核验' in log for log in logs))

    def test_failed_download_preserves_existing_file_and_hides_exception(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / data.DATA_FILES['mbpp']
            path.parent.mkdir(parents=True)
            path.write_bytes(b'existing invalid file')
            with patch.object(data.urllib.request, 'urlopen', side_effect=OSError('private secret body')):
                state = data.prepare_code_data(tmp)
            self.assertEqual(path.read_bytes(), b'existing invalid file')
            self.assertIn('检查网络', state['mbpp']['reason'])
            self.assertNotIn('private secret body', str(state))

    def test_symlink_path_cannot_read_or_write_outside_project(self):
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as external:
            (Path(tmp) / 'data').symlink_to(external, target_is_directory=True)
            state = data.code_data_status(tmp)
            self.assertIn('符号链接', state['mbpp']['reason'])
            with self.assertRaisesRegex(data.CodeDataError, '符号链接'):
                data._atomic_write(tmp, 'mbpp', b'anything')
            self.assertEqual(list(Path(external).iterdir()), [])

    def test_dataset_strings_are_never_executed_on_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            marker = Path(tmp) / 'execution-marker'
            rows = data._jsonl_records(cn_fixture())
            rows[0]['test'] = f"open({str(marker)!r}, 'w').write('executed')"
            path = Path(tmp) / data.DATA_FILES['humaneval_cn']
            path.parent.mkdir(parents=True)
            path.write_bytes(jsonl(rows))
            self.assertTrue(data.code_data_status(tmp)['humaneval_cn']['ready'])
            self.assertFalse(marker.exists())


if __name__ == '__main__':
    unittest.main()
