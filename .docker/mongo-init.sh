#!/bin/bash

# MongoDB Replica Set Initialization Script
# This ensures the replica set is properly configured with container names

MONGO_PORT=${MONGO_PORT:-30001}
REPLICA_SET_NAME="my-replica-set"

# Function to check if replica set is properly configured
check_replica_set() {
    result=$(mongo --port $MONGO_PORT --quiet --eval "
        try {
            var status = rs.status();
            if (status.ok === 1) {
                var members = status.members;
                var correctMembers = 0;
                for (var i = 0; i < members.length; i++) {
                    if (members[i].name.indexOf('mongo') === 0) {
                        correctMembers++;
                    }
                }
                if (correctMembers === 3) {
                    print('CONFIGURED_CORRECTLY');
                } else {
                    print('NEEDS_RECONFIGURATION');
                }
            } else {
                print('NOT_INITIALIZED');
            }
        } catch (e) {
            print('NOT_INITIALIZED');
        }
    ")
    echo $result
}

# Function to initialize replica set
initialize_replica_set() {
    mongo --port $MONGO_PORT --quiet --eval "
        rs.initiate({
            _id: '$REPLICA_SET_NAME',
            members: [
                {_id: 0, host: 'mongo1:30001'},
                {_id: 1, host: 'mongo2:30002'},
                {_id: 2, host: 'mongo3:30003'}
            ]
        })
    "
}

# Function to reconfigure replica set
reconfigure_replica_set() {
    mongo --port $MONGO_PORT --quiet --eval "
        rs.reconfig({
            _id: '$REPLICA_SET_NAME',
            members: [
                {_id: 0, host: 'mongo1:30001'},
                {_id: 1, host: 'mongo2:30002'},
                {_id: 2, host: 'mongo3:30003'}
            ]
        }, {force: true})
    "
}

# Main logic
status=$(check_replica_set)

case $status in
    "CONFIGURED_CORRECTLY")
        echo "Replica set is properly configured"
        exit 0
        ;;
    "NOT_INITIALIZED")
        echo "Initializing replica set..."
        initialize_replica_set
        sleep 5
        # Check again after initialization
        if [ "$(check_replica_set)" = "CONFIGURED_CORRECTLY" ]; then
            echo "Replica set initialized successfully"
            exit 0
        else
            echo "Failed to initialize replica set"
            exit 1
        fi
        ;;
    "NEEDS_RECONFIGURATION")
        echo "Reconfiguring replica set with correct hostnames..."
        reconfigure_replica_set
        sleep 5
        # Check again after reconfiguration
        if [ "$(check_replica_set)" = "CONFIGURED_CORRECTLY" ]; then
            echo "Replica set reconfigured successfully"
            exit 0
        else
            echo "Failed to reconfigure replica set"
            exit 1
        fi
        ;;
    *)
        echo "Unknown replica set status: $status"
        exit 1
        ;;
esac 