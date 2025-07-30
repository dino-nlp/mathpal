#!/usr/bin/env python3
"""
MongoDB Database Analyzer - All-in-one script
Phân tích database MongoDB 'mathpal' qua replica set

LƯU Ý:
- Script này chạy từ host machine (không phải trong Docker container)
- Sử dụng localhost:30001,30002,30003 để kết nối qua port mapping
- Replica set bên trong containers sử dụng mongo1:30001,mongo2:30002,mongo3:30003
- Warning về "name resolution" là bình thường và expected behavior
"""

import json
import time
from datetime import datetime
from typing import Dict, Any, List

from bson import ObjectId
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError


class MongoDBAnalyzer:
    """Class tổng hợp để phân tích MongoDB database qua replica set"""
    
    def __init__(self):
        # Connection string theo yêu cầu (sử dụng localhost vì script chạy từ host machine)
        self.connection_string = "mongodb://localhost:30001,localhost:30002,localhost:30003/?replicaSet=my-replica-set"
        self.database_name = "mathpal"
        self.client = None
        self.db = None
        self.connection_method = None  # Track how we connected successfully
        
    def connect(self) -> bool:
        """Kết nối đến MongoDB replica set với fallback options"""
        print(f"🔄 Đang kết nối đến MongoDB...")
        print(f"🔗 Connection string: {self.connection_string}")
        print(f"🗄️ Database: {self.database_name}")
        
        # Thử kết nối với các read preferences khác nhau
        read_preferences = [
            'secondaryPreferred',  # Ưu tiên secondary, fallback primary
            'primaryPreferred',    # Ưu tiên primary, fallback secondary  
            'secondary',           # Chỉ đọc từ secondary
            'nearest'              # Đọc từ node gần nhất
        ]
        
        for read_pref in read_preferences:
            try:
                print(f"  🔄 Thử với read preference: {read_pref}")
                
                self.client = MongoClient(
                    self.connection_string,
                    serverSelectionTimeoutMS=8000,
                    connectTimeoutMS=5000,
                    readPreference=read_pref
                )
                
                # Test connection
                self.client.admin.command('ping')
                self.db = self.client[self.database_name]
                
                self.connection_method = f"replica_set_with_{read_pref}"
                print(f"✅ Kết nối thành công với read preference: {read_pref}")
                print("-" * 80)
                return True
                
            except Exception as e:
                error_msg = str(e)
                if "name resolution" in error_msg.lower():
                    print(f"  ⚠️ {read_pref}: Không thể resolve container names từ host (bình thường)")
                else:
                    print(f"  ❌ Thất bại với {read_pref}: {error_msg[:100]}...")
                continue
        
        # Nếu replica set fail, thử kết nối trực tiếp đến từng node
        print(f"\n🔄 Replica set connection thất bại từ host machine")
        print(f"💡 Lý do: Host machine không thể resolve container names (mongo1, mongo2, mongo3)")
        print(f"🔄 Thử kết nối trực tiếp qua localhost ports...")
        
        ports = [30001, 30002, 30003]
        for port in ports:
            try:
                direct_conn = f"mongodb://localhost:{port}/?directConnection=true"
                print(f"  🔄 Thử kết nối trực tiếp: localhost:{port}")
                
                self.client = MongoClient(
                    direct_conn,
                    serverSelectionTimeoutMS=5000,
                    connectTimeoutMS=5000
                )
                
                self.client.admin.command('ping')
                self.db = self.client[self.database_name]
                
                self.connection_method = f"direct_connection_port_{port}"
                print(f"✅ Kết nối trực tiếp thành công đến localhost:{port}")
                print("-" * 80)
                return True
                
            except Exception as e:
                print(f"  ❌ Kết nối trực tiếp thất bại: {str(e)[:100]}...")
                continue
        
        print("❌ Không thể kết nối đến MongoDB!")
        print("💡 Gợi ý kiểm tra:")
        print("   - Containers có chạy không: docker ps | grep mongo")
        print("   - Ports có được exposed không: docker ps | grep '30001\\|30002\\|30003'")
        print("   - Restart containers: docker-compose restart mongo1 mongo2 mongo3")
        print("   - Hoặc restart toàn bộ: docker-compose down && docker-compose up -d")
        return False
    
    def check_replica_set_status(self) -> Dict[str, Any]:
        """Kiểm tra trạng thái replica set"""
        try:
            status = self.client.admin.command("replSetGetStatus")
            
            print("🔍 REPLICA SET STATUS:")
            print(f"  • Set name: {status.get('set', 'N/A')}")
            print(f"  • My state: {status.get('myState', 'N/A')}")
            
            state_map = {
                0: "STARTUP", 1: "PRIMARY", 2: "SECONDARY", 3: "RECOVERING",
                5: "STARTUP2", 6: "UNKNOWN", 7: "ARBITER", 8: "DOWN", 
                9: "ROLLBACK", 10: "REMOVED"
            }
            
            print("  • Members:")
            primary_found = False
            for member in status.get('members', []):
                state_num = member.get('state', 0)
                state_name = state_map.get(state_num, f"UNKNOWN({state_num})")
                health = member.get('health', 0)
                
                status_icon = "✅" if health == 1 else "❌"
                print(f"    {status_icon} {member.get('name', 'N/A')}: {state_name}")
                
                if state_num == 1:
                    primary_found = True
            
            if not primary_found:
                print("  ⚠️ Cảnh báo: Không có PRIMARY node!")
            
            return status
            
        except Exception as e:
            print(f"⚠️ Không thể lấy replica set status: {e}")
            return {}
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Lấy thống kê database"""
        try:
            stats = self.db.command("dbstats")
            return {
                "database_name": stats.get("db"),
                "collections_count": stats.get("collections", 0),
                "objects_count": stats.get("objects", 0),
                "data_size_bytes": stats.get("dataSize", 0),
                "storage_size_bytes": stats.get("storageSize", 0),
                "indexes_count": stats.get("indexes", 0),
                "index_size_bytes": stats.get("indexSize", 0),
                "avg_obj_size": stats.get("avgObjSize", 0)
            }
        except Exception as e:
            print(f"⚠️ Không thể lấy database stats: {e}")
            return {}
    
    def list_collections(self) -> List[str]:
        """Liệt kê collections"""
        try:
            collections = self.db.list_collection_names()
            return sorted(collections)
        except Exception as e:
            print(f"⚠️ Không thể liệt kê collections: {e}")
            return []
    
    def analyze_collection(self, collection_name: str) -> Dict[str, Any]:
        """Phân tích chi tiết collection"""
        try:
            collection = self.db[collection_name]
            
            # Đếm documents
            doc_count = collection.count_documents({})
            
            # Sample document
            sample_doc = collection.find_one()
            
            # Indexes
            indexes = list(collection.list_indexes())
            
            # Collection stats
            stats = {}
            try:
                stats = self.db.command("collstats", collection_name)
            except:
                pass
            
            return {
                "name": collection_name,
                "document_count": doc_count,
                "storage_size_bytes": stats.get("storageSize", 0),
                "avg_obj_size": stats.get("avgObjSize", 0),
                "total_index_size": stats.get("totalIndexSize", 0),
                "indexes": [{"name": idx.get("name"), "keys": dict(idx.get("key", {}))} for idx in indexes],
                "sample_document": self._clean_document(sample_doc) if sample_doc else None
            }
            
        except Exception as e:
            print(f"⚠️ Lỗi phân tích collection '{collection_name}': {e}")
            return {"name": collection_name, "error": str(e)}
    
    def _clean_document(self, doc: Dict) -> Dict:
        """Làm sạch document để hiển thị"""
        if not doc:
            return {}
        
        cleaned = {}
        for key, value in doc.items():
            if isinstance(value, ObjectId):
                cleaned[key] = f"ObjectId('{str(value)}')"
            elif isinstance(value, dict):
                cleaned[key] = self._clean_document(value)
            elif isinstance(value, list):
                if len(value) > 3:
                    cleaned[key] = value[:3] + [f"... và {len(value)-3} items nữa"]
                else:
                    cleaned[key] = [self._clean_document(item) if isinstance(item, dict) else item for item in value]
            elif isinstance(value, str) and len(value) > 150:
                cleaned[key] = value[:150] + "..."
            else:
                cleaned[key] = value
        
        return cleaned
    
    def format_bytes(self, bytes_value: int) -> str:
        """Format bytes thành human readable"""
        if bytes_value == 0:
            return "0 B"
        
        units = ["B", "KB", "MB", "GB", "TB"]
        unit_index = 0
        size = float(bytes_value)
        
        while size >= 1024 and unit_index < len(units) - 1:
            size /= 1024
            unit_index += 1
        
        return f"{size:.2f} {units[unit_index]}"
    
    def run_analysis(self):
        """Chạy phân tích đầy đủ"""
        print("🎯 MONGODB DATABASE ANALYZER")
        print("=" * 80)
        print(f"⏰ Thời gian: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🔗 Target: Database '{self.database_name}' trên replica set")
        print("=" * 80)
        
        # 1. Kết nối
        if not self.connect():
            return
        
        print(f"🔗 Connection method: {self.connection_method}")
        print(f"📍 Đang phân tích từ host machine đến Docker containers")
        
        # 2. Kiểm tra replica set
        print(f"\n{'='*25} REPLICA SET {'='*25}")
        self.check_replica_set_status()
        
        # 3. Database stats
        print(f"\n{'='*25} DATABASE STATS {'='*23}")
        db_stats = self.get_database_stats()
        if db_stats:
            print(f"📊 Database: {db_stats.get('database_name', 'N/A')}")
            print(f"📚 Collections: {db_stats.get('collections_count', 0)}")
            print(f"📄 Total documents: {db_stats.get('objects_count', 0):,}")
            print(f"💾 Data size: {self.format_bytes(db_stats.get('data_size_bytes', 0))}")
            print(f"🗂️ Storage size: {self.format_bytes(db_stats.get('storage_size_bytes', 0))}")
            print(f"🔍 Total indexes: {db_stats.get('indexes_count', 0)}")
            print(f"📏 Avg document size: {self.format_bytes(int(db_stats.get('avg_obj_size', 0)))}")
        
        # 4. Collections analysis
        print(f"\n{'='*25} COLLECTIONS {'='*26}")
        collections = self.list_collections()
        
        if not collections:
            print("📭 Không có collections nào trong database")
            return
        
        print(f"📋 Tìm thấy {len(collections)} collection(s):")
        for i, coll_name in enumerate(collections, 1):
            print(f"  {i}. {coll_name}")
        
        print(f"\n{'='*25} CHI TIẾT COLLECTIONS {'='*20}")
        for collection_name in collections:
            print(f"\n🔍 COLLECTION: {collection_name}")
            print("-" * 50)
            
            analysis = self.analyze_collection(collection_name)
            
            if "error" in analysis:
                print(f"❌ Lỗi: {analysis['error']}")
                continue
            
            # Basic stats
            print(f"📊 Documents: {analysis.get('document_count', 0):,}")
            print(f"💾 Storage: {self.format_bytes(analysis.get('storage_size_bytes', 0))}")
            print(f"📏 Avg size: {self.format_bytes(int(analysis.get('avg_obj_size', 0)))}")
            print(f"🗂️ Index size: {self.format_bytes(analysis.get('total_index_size', 0))}")
            
            # Indexes
            indexes = analysis.get('indexes', [])
            print(f"🔍 Indexes ({len(indexes)}):")
            for idx in indexes:
                print(f"   • {idx['name']}: {idx['keys']}")
            
            # Sample document
            sample = analysis.get('sample_document')
            if sample:
                print(f"📄 Sample document:")
                print(f"   {json.dumps(sample, indent=3, ensure_ascii=False)}")
            else:
                print(f"📄 Sample: (Collection trống)")
        
        print(f"\n{'='*80}")
        print("✅ Hoàn thành phân tích database!")
        print("🔒 Đóng kết nối...")
        
        if self.client:
            self.client.close()


def main():
    """Main function"""
    analyzer = MongoDBAnalyzer()
    try:
        analyzer.run_analysis()
    except KeyboardInterrupt:
        print("\n⚠️ Dừng phân tích theo yêu cầu người dùng")
    except Exception as e:
        print(f"\n❌ Lỗi không mong muốn: {e}")
    finally:
        if analyzer.client:
            analyzer.client.close()


if __name__ == "__main__":
    main() 