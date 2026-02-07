#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include <sys/socket.h>

// --- ΣΥΝΑΡΤΗΣΗ SEND LIST (Δέχεται λίστα από arrays) ---
static PyObject* c_send_list_of_arrays(PyObject* self, PyObject* args) {
    int sockfd;
    PyObject *list_obj;

    // 1. Διαβάζουμε: Socket ID και Λίστα (O!)
    if (!PyArg_ParseTuple(args, "iO!", &sockfd, &PyList_Type, &list_obj)) {
        return NULL;
    }

    // 2. Υπολογίζουμε το ΣΥΝΟΛΙΚΟ μέγεθος σε bytes (για το Header)
    // Πρέπει να αθροίσουμε τα bytes όλων των πινάκων στη λίστα
    Py_ssize_t num_items = PyList_Size(list_obj);
    long total_bytes = 0;

    for (Py_ssize_t i = 0; i < num_items; i++) {
        PyObject *item = PyList_GetItem(list_obj, i);
        // Έλεγχος αν είναι όντως Array
        if (!PyArray_Check(item)) {
            PyErr_SetString(PyExc_TypeError, "List elements must be numpy arrays");
            return NULL;
        }
        total_bytes += PyArray_NBYTES((PyArrayObject*)item);
    }

    // 3. Στέλνουμε το Header (Συνολικό Μέγεθος)
    // Απελευθερώνουμε το GIL για το δίκτυο
    int error_occurred = 0;
    Py_BEGIN_ALLOW_THREADS
    if (send(sockfd, &total_bytes, sizeof(total_bytes), 0) < 0) {
        error_occurred = 1;
    }
    Py_END_ALLOW_THREADS

    if (error_occurred) {
        PyErr_SetString(PyExc_IOError, "Failed to send header");
        return NULL;
    }

    // 4. Loop για αποστολή των δεδομένων κάθε πίνακα
    for (Py_ssize_t i = 0; i < num_items; i++) {
        // Παίρνουμε τον πίνακα (Χρειάζεται GIL εδώ)
        PyArrayObject *arr = (PyArrayObject*)PyList_GetItem(list_obj, i);
        
        // Παίρνουμε τον Pointer και το μέγεθος
        void *data_ptr = PyArray_DATA(arr);
        npy_intp chunk_bytes = PyArray_NBYTES(arr);
        
        ssize_t total_sent = 0;
        ssize_t sent = 0;

        // Απελευθερώνουμε το GIL για την αποστολή αυτού του κομματιού
        Py_BEGIN_ALLOW_THREADS
        while (total_sent < chunk_bytes) {
            sent = send(sockfd, (char*)data_ptr + total_sent, chunk_bytes - total_sent, 0);
            if (sent < 0) {
                error_occurred = 1;
                break;
            }
            total_sent += sent;
        }
        Py_END_ALLOW_THREADS

        if (error_occurred) {
            PyErr_SetString(PyExc_IOError, "Socket error during send loop");
            return NULL;
        }
    }

    Py_RETURN_NONE;
}

// --- ΣΥΝΑΡΤΗΣΗ RECV (Fixed for socket timeouts) ---
#include <errno.h>

static PyObject* c_recv_into_array(PyObject* self, PyObject* args) {
    int sockfd;
    PyArrayObject *arr;

    if (!PyArg_ParseTuple(args, "iO!", &sockfd, &PyArray_Type, &arr)) {
        return NULL;
    }

    void *data_ptr = PyArray_DATA(arr);
    npy_intp num_bytes = PyArray_NBYTES(arr);
    
    long msg_len = 0;
    ssize_t header_received = 0;
    ssize_t total_received = 0;
    ssize_t r = 0;
    int error_type = 0;

    Py_BEGIN_ALLOW_THREADS
    
    // Read header with timeout retry
    while (header_received < sizeof(long)) {
        r = recv(sockfd, (char*)&msg_len + header_received, sizeof(long) - header_received, 0);
        if (r > 0) {
            header_received += r;
        } else if (r == 0) {
            // Connection closed
            error_type = 1;
            break;
        } else {
            // r < 0: check errno
            if (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR) {
                // Timeout or interrupted - retry
                continue;
            } else {
                // Real error
                error_type = 1;
                break;
            }
        }
    }

    if (error_type == 0) {
        if (msg_len != num_bytes) {
            error_type = 2;
        } else {
            // Read payload with timeout retry
            while (total_received < num_bytes) {
                r = recv(sockfd, (char*)data_ptr + total_received, num_bytes - total_received, 0);
                if (r > 0) {
                    total_received += r;
                } else if (r == 0) {
                    // Connection closed
                    error_type = 3;
                    break;
                } else {
                    // r < 0: check errno
                    if (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR) {
                        // Timeout or interrupted - retry
                        continue;
                    } else {
                        // Real error
                        error_type = 3;
                        break;
                    }
                }
            }
        }
    }
    Py_END_ALLOW_THREADS

    if (error_type == 1) Py_RETURN_FALSE;
    if (error_type == 2) { PyErr_Format(PyExc_ValueError, "Size mismatch: %ld vs %ld", num_bytes, msg_len); return NULL; }
    if (error_type == 3) { PyErr_SetString(PyExc_EOFError, "Connection broken"); return NULL; }

    Py_RETURN_TRUE;
}

static PyMethodDef FastNetMethods[] = {
    // Προσοχή: Αλλάξαμε το όνομα της συνάρτησης εδώ
    {"send_list", c_send_list_of_arrays, METH_VARARGS, "Send list of numpy arrays sequentially"},
    {"recv_into_array", c_recv_into_array, METH_VARARGS, "Recv socket data directly into array"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef fastnetmodule = {
    PyModuleDef_HEAD_INIT, "fast_net", NULL, -1, FastNetMethods
};

PyMODINIT_FUNC PyInit_fast_net(void) {
    import_array();
    return PyModule_Create(&fastnetmodule);
}